import asyncio
import itertools
import logging
from functools import partial

from transitions.core import Condition, Transition, State, MachineError, Machine, Event


_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class AsyncCondition(Condition):
    """ A helper class to call async condition checks in the intended way."""

    async def check(self, event_data):
        """ Check whether the condition passes.
        Args:
            event_data (EventData): An EventData instance to pass to the
                condition (if event sending is enabled) or to extract arguments
                from (if event sending is disabled). Also contains the data
                model attached to the current machine which is used to invoke
                the condition.
        """
        condition_check = self.get_condition_check(event_data)
        if asyncio.iscoroutine(condition_check):
            condition_check = await condition_check
        return condition_check == self.target


class AsyncState(State):
    """A persistent representation of a state managed by a ``AsyncMachine``."""

    async def enter(self, event_data):
        """ Triggered when a state is entered. """
        _LOGGER.debug("%sEntering state %s. Processing callbacks...", event_data.machine.name, self.name)
        for handle in self.on_enter:
            await event_data.machine.callback(handle, event_data)
        _LOGGER.info("%sEntered state %s", event_data.machine.name, self.name)

    async def exit(self, event_data):
        """ Triggered when a state is exited. """
        _LOGGER.debug("%sExiting state %s. Processing callbacks...", event_data.machine.name, self.name)
        for handle in self.on_exit:
            await event_data.machine.callback(handle, event_data)
        _LOGGER.info("%sExited state %s", event_data.machine.name, self.name)


class AsyncTransitionMixin:
    """ Representation of a async transition managed by a ``AsyncMachine`` instance."""

    condition_class = AsyncCondition

    async def execute(self, event_data):
        """ Execute the transition.
        Args:
            event_data: An instance of class EventData.
        Returns: boolean indicating whether or not the transition was
            successfully executed (True if successful, False if not).
        """
        _LOGGER.debug("%sInitiating transition from state %s to state %s...",
                      event_data.machine.name, self.source, self.dest)
        machine = event_data.machine

        for func in self.prepare:
            await machine.callback(func, event_data)
            _LOGGER.debug("Executed callback '%s' before conditions.", func)

        for cond in self.conditions:
            if not await cond.check(event_data):
                _LOGGER.debug("%sTransition condition failed: %s() does not return %s. Transition halted.",
                              event_data.machine.name, cond.func, cond.target)
                return False
        for func in itertools.chain(machine.before_state_change, self.before):
            await machine.callback(func, event_data)
            _LOGGER.debug("%sExecuted callback '%s' before transition.", event_data.machine.name, func)

        if self.dest:
            await self._change_state(event_data)

        for func in itertools.chain(self.after, machine.after_state_change):
            await machine.callback(func, event_data)
            _LOGGER.debug("%sExecuted callback '%s' after transition.", event_data.machine.name, func)
        return True

    async def _change_state(self, event_data):
        await event_data.machine.get_state(self.source).exit(event_data)
        event_data.machine.set_state(self.dest, event_data.model)
        event_data.update(event_data.model)
        await event_data.machine.get_state(self.dest).enter(event_data)


class AsyncTransition(AsyncTransitionMixin, Transition):
    pass


class AsyncEventMixin:
    """ A collection of transitions assigned to the same trigger"""

    async def trigger(self, model, *args, **kwargs):
        func = partial(self._trigger, model, *args, **kwargs)
        # pylint: disable=protected-access
        # noinspection PyProtectedMember
        # Machine._process should not be called somewhere else. That's why it should not be exposed
        # to Machine users.
        return await self.machine._process(func)

    async def _trigger(self, model, *args, **kwargs):
        """ Internal trigger function called by the ``Machine`` instance. This should not
        be called directly but via the public method ``Machine.trigger``.
        """
        event_data = self._prepare_event_data(model, *args, **kwargs)
        if event_data:
            return await self._process(event_data)
        else:
            return False

    async def _process(self, event_data):
        for func in self.machine.prepare_event:
            await self.machine.callback(func, event_data)
            _LOGGER.debug("Executed machine preparation callback '%s' before conditions.", func)

        try:
            for trans in self.transitions[event_data.state.name]:
                event_data.transition = trans
                transition_result = await trans.execute(event_data)

                if transition_result:
                    event_data.result = True
                    break
        except Exception as e:
            event_data.error = e
            raise
        finally:
            for func in self.machine.finalize_event:
                await self.machine.callback(func, event_data)
                _LOGGER.debug("Executed machine finalize callback '%s'.", func)
        return event_data.result


class AsyncEvent(AsyncEventMixin, Event):
    pass


class AsyncMachineMixin:

    async def callback(self, func, event_data):

        callback = super().callback(func, event_data)
        if asyncio.iscoroutine(callback):
            await callback

    async def _process(self, trigger):
        # default processing
        if not self.has_queue:
            if not self._transition_queue:
                # if trigger raises an Error, it has to be handled by the Machine.process caller
                return await trigger()
            else:
                raise MachineError(
                    "Attempt to process events synchronously while transition queue is not empty!"
                )

        # process queued events
        self._transition_queue.append(trigger)
        # another entry in the queue implies a running transition; skip immediate execution
        if len(self._transition_queue) > 1:
            return True

        # execute as long as transition queue is not empty
        while self._transition_queue:
            try:
                callback = self._transition_queue[0]()

                if asyncio.iscoroutine(callback):
                    await callback

                self._transition_queue.popleft()
            except Exception:
                # if a transition raises an exception, clear queue and delegate exception handling
                self._transition_queue.clear()
                raise
        return True

    async def dispatch(self, trigger, *args, **kwargs):
        """ Trigger an event on all models assigned to the machine.
        Args:
            trigger (str): Event name
            *args (list): List of arguments passed to the event trigger
            **kwargs (dict): Dictionary of keyword arguments passed to the event trigger
        Returns:
            bool The truth value of all triggers combined with AND
        """
        return all([await getattr(model, trigger)(*args, **kwargs) for model in self.models])


class AsyncMachine(AsyncMachineMixin, Machine):
    """ Machine manages states, transitions and models. In case it is initialized without a specific model
    (or specifically no model), it will also act as a model itself. Machine takes also care of decorating
    models with conveniences functions related to added transitions and states during runtime."""

    state_cls = AsyncState
    transition_cls = AsyncTransition
    event_cls = AsyncEvent
