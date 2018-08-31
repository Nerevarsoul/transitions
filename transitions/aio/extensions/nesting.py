import logging
from functools import partial

from transitions.core import listify
from transitions.extensions.nesting import (NestedState, NestedTransition, NestedEvent,
                                            HierarchicalMachine)
from ..core import AsyncTransitionMixin, AsyncEventMixin, AsyncMachineMixin


_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class AsyncNestedState(NestedState):

    async def exit_nested(self, event_data, target_state):
        """ Tracks child states to exit when the states is exited itself. This should not
            be triggered by the user but will be handled by the hierarchical machine.
        Args:
            event_data (EventData): Event related data.
            target_state (NestedState): The state to be entered.

        Returns: int level of the currently investigated (sub)state.

        """
        if self == target_state:
            await self.exit(event_data)
            return self.level
        elif self.level > target_state.level:
            await self.exit(event_data)
            return await self.parent.exit_nested(event_data, target_state)
        elif self.level <= target_state.level:
            tmp_state = target_state
            while self.level != tmp_state.level:
                tmp_state = tmp_state.parent
            tmp_self = self
            while tmp_self.level > 0 and tmp_state.parent.name != tmp_self.parent.name:
                tmp_self.exit(event_data)
                tmp_self = tmp_self.parent
                tmp_state = tmp_state.parent
            if tmp_self == tmp_state:
                return tmp_self.level + 1
            await tmp_self.exit(event_data)
            return tmp_self.level

    async def enter_nested(self, event_data, level=None):
        """ Tracks parent states to be entered when the states is entered itself. This should not
            be triggered by the user but will be handled by the hierarchical machine.
        Args:
            event_data (EventData): Event related data.
            level (int): The level of the currently entered parent.
        """
        if level is not None and level <= self.level:
            if level != self.level:
                self.parent.enter_nested(event_data, level)
            await self.enter(event_data)


class AsyncNestedTransition(AsyncTransitionMixin, NestedTransition):
    async def execute(self, event_data):
        """ Extends transitions.core.transitions to handle nested states. """
        if self.dest is None:
            return await super().execute(event_data)
        dest_state = event_data.machine.get_state(self.dest)
        while dest_state.initial:
            dest_state = event_data.machine.get_state(dest_state.initial)
        self.dest = dest_state.name
        return await super().execute(event_data)

    # The actual state change method 'execute' in Transition was restructured to allow overriding
    async def _change_state(self, event_data):
        machine = event_data.machine
        model = event_data.model
        dest_state = machine.get_state(self.dest)
        source_state = machine.get_state(model.state)
        lvl = await source_state.exit_nested(event_data, dest_state)
        event_data.machine.set_state(self.dest, model)
        event_data.update(model)
        await dest_state.enter_nested(event_data, lvl)


class AsyncNestedEvent(AsyncEventMixin, NestedEvent):
    pass


class AsyncHierarchicalMachine(AsyncMachineMixin, HierarchicalMachine):
    state_cls = AsyncNestedState
    transition_cls = AsyncNestedTransition
    event_cls = AsyncNestedEvent

    async def to_state(self, model, state_name, *args, **kwargs):
        """ Helper function to add go to states in case a custom state separator is used.
        Args:
            model (class): The model that should be used.
            state_name (str): Name of the destination state.
        """

        event, state_name = self.get_event(model, state_name, *args, **kwargs)
        await self._create_transition(model.state, state_name).execute(event)

    async def add_model(self, model, initial=None):
        """ Extends transitions.core.Machine.add_model by applying a custom 'to' function to
            the added model.
        """
        super().add_model(model, initial=initial)
        models = listify(model)
        for mod in models:
            mod = self if mod == 'self' else mod
            # TODO: Remove 'mod != self' in 0.7.0
            if hasattr(mod, 'to') and mod != self:
                _LOGGER.warning("%sModel already has a 'to'-method. It will NOT "
                                "be overwritten by NestedMachine", self.name)
            else:
                to_func = partial(self.to_state, mod)
                await setattr(mod, 'to', to_func)
