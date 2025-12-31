# Copyright 2016 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import pydoc
from functools import partial

from kubernetes import client
from .api_client import ApiClient
from .stream import WsApiClient

PYDOC_RETURN_LABEL = ":return:"
PYDOC_FOLLOW_PARAM = ":param follow:"

# Removing this suffix from return type name should give us event's object
# type. e.g., if list_namespaces() returns "NamespaceList" type,
# then list_namespaces(watch=true) returns a stream of events with objects
# of type "Namespace". In case this assumption is not true, user should
# provide return_type to Watch class's __init__.
TYPE_LIST_SUFFIX = "List"

class SimpleNamespace:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _find_return_type(func):
    for line in pydoc.getdoc(func).splitlines():
        if line.startswith(PYDOC_RETURN_LABEL):
            return line[len(PYDOC_RETURN_LABEL):].strip()
    return ""


class BaseWatch(object):
    """
    Base class for Kubernetes API resource streaming.

    Provides common functionality for both HTTP chunked streaming (Watch)
    and WebSocket streaming (Stream).
    """

    def __init__(self, api_client, return_type=None):
        self._raw_return_type = return_type
        self._stop = False
        self._api_client = api_client
        self.resource_version = None

    def stop(self):
        self._stop = True

    def get_return_type(self, func):
        if self._raw_return_type:
            return self._raw_return_type
        return_type = _find_return_type(func)
        if return_type.endswith(TYPE_LIST_SUFFIX):
            return return_type[:-len(TYPE_LIST_SUFFIX)]
        return return_type

    def get_watch_argument_name(self, func):
        if PYDOC_FOLLOW_PARAM in pydoc.getdoc(func):
            return 'follow'
        else:
            return 'watch'

    def unmarshal_event(self, data, return_type):
        if not data or data.isspace():
            return None
        try:
            js = json.loads(data)
            js['raw_object'] = js['object']
            # BOOKMARK event is treated the same as ERROR for a quick fix of
            # decoding exception
            # TODO: make use of the resource_version in BOOKMARK event for more
            # efficient WATCH
            if return_type and js['type'] != 'ERROR' and js['type'] != 'BOOKMARK':
                obj = SimpleNamespace(data=json.dumps(js['raw_object']))
                js['object'] = self._api_client.deserialize(obj, return_type)
                if hasattr(js['object'], 'metadata'):
                    self.resource_version = js['object'].metadata.resource_version
                # For custom objects that we don't have model defined, json
                # deserialization results in dictionary
                elif (isinstance(js['object'], dict) and 'metadata' in js['object']
                      and 'resourceVersion' in js['object']['metadata']):
                    self.resource_version = js['object']['metadata'][
                        'resourceVersion']
            return js
        except json.JSONDecodeError:
            return None

    def stream(self, func, *args, **kwargs):
        """Stream an API resource and return results via a generator.

        :param func: The API function pointer. Any parameter to the function
                     can be passed after this parameter.

        :return: Event object with these keys:
                   'type': The type of event such as "ADDED", "DELETED", etc.
                   'raw_object': a dict representing the watched object.
                   'object': A model representation of raw_object. The name of
                             model will be determined based on
                             the func's doc string. If it cannot be determined,
                             'object' value will be the same as 'raw_object'.
        """
        self._stop = False
        self.return_type = self.get_return_type(func)
        kwargs[self.get_watch_argument_name(func)] = True
        kwargs['_preload_content'] = False
        if 'resource_version' in kwargs:
            self.resource_version = kwargs['resource_version']

        self.func = partial(func, *args, **kwargs)

        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await self.next()
        except:  # noqa: E722
            await self.close()
            raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    # Abstract methods to be implemented by subclasses
    async def next(self):
        """Get the next event. Must be implemented by subclasses."""
        raise NotImplementedError

    async def close(self):
        """Close the connection. Must be implemented by subclasses."""
        raise NotImplementedError


class Stream(BaseWatch):
    """
    WebSocket-based streaming for Kubernetes API resources.

    Similar to Watch but uses WebSocket connections via WsApiClient
    instead of chunked HTTP streaming.
    """

    def __init__(self, api_client=None, return_type=None):
        if api_client is None:
            api_client = WsApiClient()
        super().__init__(api_client, return_type)
        self.ws_ctx = None  # WebSocket context manager
        self.ws = None      # Actual WebSocket connection

    def stop(self):
        self._stop = True

    async def _reconnect(self):
        if self.ws is not None:
            await self.ws.close()
            self.ws = None
        if self.ws_ctx is not None:
            await self.ws_ctx.__aexit__(None, None, None)
            self.ws_ctx = None
        if self.resource_version:
            self.func.keywords['resource_version'] = self.resource_version

    async def next(self):
        watch_forever = 'timeout_seconds' not in self.func.keywords
        retry_410 = watch_forever

        while True:
            # Set the websocket connection to the user supplied function (eg
            # `list_namespaced_pods`) if this is the first iteration.
            if self.ws is None:
                # func() returns a context manager, we need to enter it
                self.ws_ctx = await self.func()
                self.ws = await self.ws_ctx.__aenter__()

            # Abort at the current iteration if the user has called `stop` on this
            # stream instance.
            if self._stop:
                raise StopAsyncIteration

            # Fetch the next K8s response from WebSocket.
            try:
                msg = await self.ws.receive()

                # Handle WebSocket message types
                if msg.type == 1:  # aiohttp.WSMsgType.TEXT
                    line = msg.data
                elif msg.type == 8:  # aiohttp.WSMsgType.CLOSE
                    # WebSocket closed
                    if watch_forever:
                        await self._reconnect()
                        continue
                    else:
                        raise StopAsyncIteration
                elif msg.type == 9:  # aiohttp.WSMsgType.ERROR
                    # WebSocket error
                    if watch_forever:
                        await self._reconnect()
                        continue
                    else:
                        raise StopAsyncIteration
                else:
                    # Skip other message types (binary, ping, pong, etc.)
                    continue

            except asyncio.TimeoutError:
                # This exception can be raised by aiohttp (client timeout)
                # but we don't retry if server side timeout is applied.
                if watch_forever:
                    await self._reconnect()
                    continue
                else:
                    raise
            except Exception as e:
                # Handle other exceptions (connection errors, etc.)
                if watch_forever:
                    await self._reconnect()
                    continue
                else:
                    raise

            # Special case for faster log streaming
            if self.return_type == 'str':
                if line == '':
                    # end of log
                    raise StopAsyncIteration
                return line

            # Stop the iterator if K8s sends an empty response. This happens when
            # e.g. the supplied timeout has expired.
            if line == '':
                if watch_forever:
                    await self._reconnect()
                    continue
                raise StopAsyncIteration

            # retry 410 error only once
            try:
                event = self.unmarshal_event(line, self.return_type)
            except client.exceptions.ApiException as ex:
                if ex.status == 410 and retry_410:
                    retry_410 = False  # retry only once
                    await self._reconnect()
                    continue
                raise
            retry_410 = watch_forever
            return event

    async def close(self):
        await self._api_client.close()
        if self.ws is not None:
            await self.ws.close()
            self.ws = None
            await self.ws_ctx.__aexit__(None, None, None)
            self.ws_ctx = None


class Watch(BaseWatch):
    """
    HTTP chunked streaming for Kubernetes API resources.

    Uses standard HTTP chunked transfer encoding via ApiClient
    for streaming API resource updates.
    """

    def __init__(self, api_client=None, return_type=None):
        if api_client is None:
            api_client = ApiClient()
        super().__init__(api_client, return_type)
        self.resp = None

    def _reconnect(self):
        self.resp.close()
        self.resp = None
        if self.resource_version:
            self.func.keywords['resource_version'] = self.resource_version

    async def next(self):
        watch_forever = 'timeout_seconds' not in self.func.keywords
        retry_410 = watch_forever

        while True:
            # Set the response object to the user supplied function (eg
            # `list_namespaced_pods`) if this is the first iteration.
            if self.resp is None:
                self.resp = await self.func()

            # Abort at the current iteration if the user has called `stop` on this
            # stream instance.
            if self._stop:
                raise StopAsyncIteration

            # Fetch the next K8s response.
            try:
                line = await self.resp.content.readline()
            except asyncio.TimeoutError:
                # This exception can be raised by aiohttp (client timeout)
                # but we don't retry if server side timeout is applied.
                if watch_forever:
                    self._reconnect()
                    continue
                else:
                    raise

            line = line.decode('utf8')

            # Special case for faster log streaming
            if self.return_type == 'str':
                if line == '':
                    # end of log
                    raise StopAsyncIteration
                return line

            # Stop the iterator if K8s sends an empty response. This happens when
            # e.g. the supplied timeout has expired.
            if line == '':
                if watch_forever:
                    self._reconnect()
                    continue
                raise StopAsyncIteration

            # retry 410 error only once
            try:
                event = self.unmarshal_event(line, self.return_type)
            except client.exceptions.ApiException as ex:
                if ex.status == 410 and retry_410:
                    retry_410 = False  # retry only once
                    self._reconnect()
                    continue
                raise
            retry_410 = watch_forever
            return event

    async def close(self):
        await self._api_client.close()
        if self.resp is not None:
            self.resp.release()
            self.resp = None
