try:
    from builtins import object
except ImportError:
    pass

import asyncio
import warnings
import sys

from .utils import AsyncStuff, AsyncInheritedStuff

from functools import partial
from transitions.aio import AsyncMachine, AsyncState
from transitions.core import listify, _prep_ordered_arg, MachineError, EventData
from unittest import TestCase, skipIf

try:
    from unittest.mock import MagicMock
except ImportError:
    from mock import MagicMock


async def on_exit_A(event):
    event.model.exit_A_called = True


async def on_exit_B(event):
    event.model.exit_B_called = True


class TestTransitions(TestCase):

    def setUp(self):
        self.stuff = AsyncStuff()
        self.loop = asyncio.get_event_loop()

    def tearDown(self):
        pass

    def test_init_machine_with_hella_arguments(self):
        states = [
            AsyncState('State1'),
            'State2',
            {
                'name': 'State3',
                'on_enter': 'hello_world'
            }
        ]
        transitions = [
            {'trigger': 'advance',
                'source': 'State2',
                'dest': 'State3'
             }
        ]
        s = AsyncStuff()
        m = AsyncMachine(model=s, states=states, transitions=transitions, initial='State2')
        self.loop.run_until_complete(s.advance())
        self.assertEqual(s.message, 'Hello World!')

    def test_transition_definitions(self):
        states = ['A', 'B', 'C', 'D']
        # Define with list of dictionaries
        transitions = [
            {'trigger': 'walk', 'source': 'A', 'dest': 'B'},
            {'trigger': 'run', 'source': 'B', 'dest': 'C'},
            {'trigger': 'sprint', 'source': 'C', 'dest': 'D'}
        ]
        m = AsyncMachine(states=states, transitions=transitions, initial='A')
        self.loop.run_until_complete(m.walk())
        self.assertEqual(m.state, 'B')
        # Define with list of lists
        transitions = [
            ['walk', 'A', 'B'],
            ['run', 'B', 'C'],
            ['sprint', 'C', 'D']
        ]
        m = AsyncMachine(states=states, transitions=transitions, initial='A')
        self.loop.run_until_complete(m.to_C())
        self.loop.run_until_complete(m.sprint())
        self.assertEqual(m.state, 'D')

    def test_transitioning(self):
        s = self.stuff
        s.machine.add_transition('advance', 'A', 'B')
        s.machine.add_transition('advance', 'B', 'C')
        s.machine.add_transition('advance', 'C', 'D')
        self.loop.run_until_complete(s.advance())
        self.assertEqual(s.state, 'B')
        self.assertFalse(s.is_A())
        self.assertTrue(s.is_B())
        self.loop.run_until_complete(s.advance())
        self.assertEqual(s.state, 'C')

    def test_pass_state_instances_instead_of_names(self):
        state_A = AsyncState('A')
        state_B = AsyncState('B')
        states = [state_A, state_B]
        m = AsyncMachine(states=states, initial=state_A)
        assert m.state == 'A'
        m.add_transition('advance', state_A, state_B)
        self.loop.run_until_complete(m.advance())
        assert m.state == 'B'
        state_B2 = AsyncState('B', on_enter='this_passes')
        with self.assertRaises(ValueError):
            m.add_transition('advance2', state_A, state_B2)
        m2 = AsyncMachine(states=states, initial=state_A.name)
        assert m.initial == m2.initial
        with self.assertRaises(ValueError):
            AsyncMachine(states=states, initial=AsyncState('A'))

    def test_conditions(self):
        s = self.stuff
        s.machine.add_transition('advance', 'A', 'B', conditions='this_passes')
        s.machine.add_transition('advance', 'B', 'C', unless=['this_fails'])
        s.machine.add_transition('advance', 'C', 'D', unless=['this_fails',
                                                              'this_passes'])
        self.loop.run_until_complete(s.advance())
        self.assertEqual(s.state, 'B')
        self.loop.run_until_complete(s.advance())
        self.assertEqual(s.state, 'C')
        self.loop.run_until_complete(s.advance())
        self.assertEqual(s.state, 'C')

    def test_conditions_with_partial(self):
        def check(result):
            return result

        s = self.stuff
        s.machine.add_transition('advance', 'A', 'B',
                                 conditions=partial(check, True))
        s.machine.add_transition('advance', 'B', 'C',
                                 unless=[partial(check, False)])
        s.machine.add_transition('advance', 'C', 'D',
                                 unless=[partial(check, False), partial(check, True)])
        self.loop.run_until_complete(s.advance())
        self.assertEqual(s.state, 'B')
        self.loop.run_until_complete(s.advance())
        self.assertEqual(s.state, 'C')
        self.loop.run_until_complete(s.advance())
        self.assertEqual(s.state, 'C')

    def test_multiple_add_transitions_from_state(self):
        s = self.stuff
        s.machine.add_transition(
            'advance', 'A', 'B', conditions=['this_fails'])
        s.machine.add_transition('advance', 'A', 'C')
        self.loop.run_until_complete(s.advance())
        self.assertEqual(s.state, 'C')

    def test_use_machine_as_model(self):
        states = ['A', 'B', 'C', 'D']
        m = AsyncMachine(states=states, initial='A')
        m.add_transition('move', 'A', 'B')
        m.add_transition('move_to_C', 'B', 'C')
        self.loop.run_until_complete(m.move())
        self.assertEqual(m.state, 'B')

    def test_state_change_listeners(self):
        s = self.stuff
        s.machine.add_transition('advance', 'A', 'B')
        s.machine.add_transition('reverse', 'B', 'A')
        s.machine.on_enter_B('hello_world')
        s.machine.on_exit_B('goodbye')
        self.loop.run_until_complete(s.advance())
        self.assertEqual(s.state, 'B')
        self.assertEqual(s.message, 'Hello World!')
        self.loop.run_until_complete(s.reverse())
        self.assertEqual(s.state, 'A')
        self.assertTrue(s.message.startswith('So long'))

    def test_before_after_callback_addition(self):
        m = AsyncMachine(AsyncStuff(), states=['A', 'B', 'C'], initial='A')
        m.add_transition('move', 'A', 'B')
        trans = m.events['move'].transitions['A'][0]
        trans.add_callback('after', 'increase_level')
        self.loop.run_until_complete(m.model.move())
        self.assertEqual(m.model.level, 2)

    def test_before_after_transition_listeners(self):
        m = AsyncMachine(AsyncStuff(), states=['A', 'B', 'C'], initial='A')
        m.add_transition('move', 'A', 'B')
        m.add_transition('move', 'B', 'C')

        m.before_move('increase_level')
        self.loop.run_until_complete(m.model.move())
        self.assertEqual(m.model.level, 2)
        self.loop.run_until_complete(m.model.move())
        self.assertEqual(m.model.level, 3)

    def test_prepare(self):
        m = AsyncMachine(AsyncStuff(), states=['A', 'B', 'C'], initial='A')
        m.add_transition('move', 'A', 'B', prepare='increase_level')
        m.add_transition('move', 'B', 'C', prepare='increase_level')
        m.add_transition('move', 'C', 'A', prepare='increase_level', conditions='this_fails')
        m.add_transition('dont_move', 'A', 'C', prepare='increase_level')

        m.prepare_move('increase_level')

        self.loop.run_until_complete(m.model.move())
        self.assertEqual(m.model.state, 'B')
        self.assertEqual(m.model.level, 3)

        self.loop.run_until_complete(m.model.move())
        self.assertEqual(m.model.state, 'C')
        self.assertEqual(m.model.level, 5)

        # State does not advance, but increase_level still runs
        self.loop.run_until_complete(m.model.move())
        self.assertEqual(m.model.state, 'C')
        self.assertEqual(m.model.level, 7)

        # An invalid transition shouldn't execute the callback
        try:
            self.loop.run_until_complete(m.model.dont_move())
        except MachineError as e:
            self.assertTrue("Can't trigger event" in str(e))

        self.assertEqual(m.model.state, 'C')
        self.assertEqual(m.model.level, 7)

    def test_state_model_change_listeners(self):
        s = self.stuff
        s.machine.add_transition('go_e', 'A', 'E')
        s.machine.add_transition('go_f', 'E', 'F')
        s.machine.on_enter_F('hello_F')
        self.loop.run_until_complete(s.go_e())
        self.assertEqual(s.state, 'E')
        self.assertEqual(s.message, 'I am E!')
        self.loop.run_until_complete(s.go_f())
        self.assertEqual(s.state, 'F')
        self.assertEqual(s.exit_message, 'E go home...')
        assert 'I am F!' in s.message
        assert 'Hello F!' in s.message

    def test_inheritance(self):
        states = ['A', 'B', 'C', 'D', 'E']
        s = AsyncInheritedStuff(states=states, initial='A')
        s.add_transition('advance', 'A', 'B', conditions='this_passes')
        s.add_transition('advance', 'B', 'C')
        s.add_transition('advance', 'C', 'D')
        self.loop.run_until_complete(s.advance())
        self.assertEqual(s.state, 'B')
        self.assertFalse(s.is_A())
        self.assertTrue(s.is_B())
        self.loop.run_until_complete(s.advance())
        self.assertEqual(s.state, 'C')

        class NewMachine(AsyncMachine):
            def __init__(self, *args, **kwargs):
                super(NewMachine, self).__init__(*args, **kwargs)

        n = NewMachine(states=states, transitions=[['advance', 'A', 'B']], initial='A')
        self.assertTrue(n.is_A())
        self.loop.run_until_complete(n.advance())
        self.assertTrue(n.is_B())
        with self.assertRaises(ValueError):
            NewMachine(state=['A', 'B'])

    def test_send_event_data_callbacks(self):
        states = ['A', 'B', 'C', 'D', 'E']
        s = AsyncStuff()
        # First pass positional and keyword args directly to the callback
        m = AsyncMachine(model=s, states=states, initial='A', send_event=False,
                         auto_transitions=True)
        m.add_transition(
            trigger='advance', source='A', dest='B', before='set_message')
        self.loop.run_until_complete(s.advance(message='Hallo. My name is Inigo Montoya.'))
        self.assertTrue(s.message.startswith('Hallo.'))
        self.loop.run_until_complete(s.to_A())
        self.loop.run_until_complete(s.advance('Test as positional argument'))
        self.assertTrue(s.message.startswith('Test as'))
        # Now wrap arguments in an EventData instance
        m.send_event = True
        m.add_transition(
            trigger='advance', source='B', dest='C', before='extract_message')
        self.loop.run_until_complete(s.advance(message='You killed my father. Prepare to die.'))
        self.assertTrue(s.message.startswith('You'))

    def test_send_event_data_conditions(self):
        states = ['A', 'B', 'C', 'D']
        s = AsyncStuff()
        # First pass positional and keyword args directly to the condition
        m = AsyncMachine(model=s, states=states, initial='A', send_event=False)
        m.add_transition(
            trigger='advance', source='A', dest='B',
            conditions='this_fails_by_default')
        self.loop.run_until_complete(s.advance(boolean=True))
        self.assertEqual(s.state, 'B')
        # Now wrap arguments in an EventData instance
        m.send_event = True
        m.add_transition(
            trigger='advance', source='B', dest='C',
            conditions='extract_boolean')
        self.loop.run_until_complete(s.advance(boolean=False))
        self.assertEqual(s.state, 'B')

    def test_auto_transitions(self):
        states = ['A', {'name': 'B'}, AsyncState(name='C')]
        m = AsyncMachine('self', states, initial='A', auto_transitions=True)
        self.loop.run_until_complete(m.to_B())
        self.assertEqual(m.state, 'B')
        self.loop.run_until_complete(m.to_C())
        self.assertEqual(m.state, 'C')
        self.loop.run_until_complete(m.to_A())
        self.assertEqual(m.state, 'A')
        # Should fail if auto transitions is off...
        m = AsyncMachine('self', states, initial='A', auto_transitions=False)
        with self.assertRaises(AttributeError):
            self.loop.run_until_complete(m.to_C())

    def test_ordered_transitions(self):
        states = ['beginning', 'middle', 'end']
        m = AsyncMachine('self', states)
        m.add_ordered_transitions()
        self.assertEqual(m.state, 'initial')
        self.loop.run_until_complete(m.next_state())
        self.assertEqual(m.state, 'beginning')
        self.loop.run_until_complete(m.next_state())
        self.loop.run_until_complete(m.next_state())
        self.assertEqual(m.state, 'end')
        self.loop.run_until_complete(m.next_state())
        self.assertEqual(m.state, 'initial')

        # Include initial state in loop
        m = AsyncMachine('self', states)
        m.add_ordered_transitions(loop_includes_initial=False)
        self.loop.run_until_complete(m.to_end())
        self.loop.run_until_complete(m.next_state())
        self.assertEqual(m.state, 'beginning')

        # Do not loop transitions
        m = AsyncMachine('self', states)
        m.add_ordered_transitions(loop=False)
        self.loop.run_until_complete(m.to_end())
        with self.assertRaises(MachineError):
            self.loop.run_until_complete(m.next_state())

        # Test user-determined sequence and trigger name
        m = AsyncMachine('self', states, initial='beginning')
        m.add_ordered_transitions(['end', 'beginning'], trigger='advance')
        self.loop.run_until_complete(m.advance())
        self.assertEqual(m.state, 'end')
        self.loop.run_until_complete(m.advance())
        self.assertEqual(m.state, 'beginning')

        # Via init argument
        m = AsyncMachine('self', states, initial='beginning', ordered_transitions=True)
        self.loop.run_until_complete(m.next_state())
        self.assertEqual(m.state, 'middle')

        # Alter initial state
        m = AsyncMachine('self', states, initial='middle', ordered_transitions=True)
        self.loop.run_until_complete(m.next_state())
        self.assertEqual(m.state, 'end')
        self.loop.run_until_complete(m.next_state())
        self.assertEqual(m.state, 'beginning')

    def test_ignore_invalid_triggers(self):
        a_state = AsyncState('A')
        transitions = [['a_to_b', 'A', 'B']]
        # Exception is triggered by default
        b_state = AsyncState('B')
        m1 = AsyncMachine('self', states=[a_state, b_state], transitions=transitions,
                          initial='B')
        with self.assertRaises(MachineError):
            self.loop.run_until_complete(m1.a_to_b())
        # Exception is suppressed, so this passes
        b_state = AsyncState('B', ignore_invalid_triggers=True)
        m2 = AsyncMachine('self', states=[a_state, b_state], transitions=transitions,
                          initial='B')
        self.loop.run_until_complete(m2.a_to_b())
        # Set for some states but not others
        new_states = ['C', 'D']
        m1.add_states(new_states, ignore_invalid_triggers=True)
        self.loop.run_until_complete(m1.to_D())
        self.loop.run_until_complete(m1.a_to_b())  # passes because exception suppressed for D
        self.loop.run_until_complete(m1.to_B())
        with self.assertRaises(MachineError):
            self.loop.run_until_complete(m1.a_to_b())
        # Set at machine level
        m3 = AsyncMachine('self', states=[a_state, b_state], transitions=transitions,
                          initial='B', ignore_invalid_triggers=True)
        self.loop.run_until_complete(m3.a_to_b())

    def test_string_callbacks(self):

        m = AsyncMachine(states=['A', 'B'],
                         before_state_change='before_state_change',
                         after_state_change='after_state_change', send_event=True,
                         initial='A', auto_transitions=True)

        m.before_state_change = MagicMock()
        m.after_state_change = MagicMock()

        self.loop.run_until_complete(m.to_B())

        self.assertTrue(m.before_state_change[0].called)
        self.assertTrue(m.after_state_change[0].called)

        # after_state_change should have been called with EventData
        event_data = m.after_state_change[0].call_args[0][0]
        self.assertIsInstance(event_data, EventData)
        self.assertTrue(event_data.result)

    def test_function_callbacks(self):
        before_state_change = MagicMock()
        after_state_change = MagicMock()

        m = AsyncMachine('self', states=['A', 'B'],
                         before_state_change=before_state_change,
                         after_state_change=after_state_change, send_event=True,
                         initial='A', auto_transitions=True)

        self.loop.run_until_complete(m.to_B())
        self.assertTrue(m.before_state_change[0].called)
        self.assertTrue(m.after_state_change[0].called)

    def test_state_callable_callbacks(self):

        class Model:

            def __init__(self):
                self.exit_A_called = False
                self.exit_B_called = False

            async def on_enter_A(self, event):
                pass

            async def on_enter_B(self, event):
                pass

        states = [AsyncState(name='A', on_enter='on_enter_A', on_exit='tests.test_core.on_exit_A'),
                  AsyncState(name='B', on_enter='on_enter_B', on_exit=on_exit_B),
                  AsyncState(name='C', on_enter='tests.test_core.AAAA')]

        model = Model()
        machine = AsyncMachine(model, states=states, send_event=True, initial='A')
        state_a = machine.get_state('A')
        state_b = machine.get_state('B')
        self.assertEqual(len(state_a.on_enter), 1)
        self.assertEqual(len(state_a.on_exit), 1)
        self.assertEqual(len(state_b.on_enter), 1)
        self.assertEqual(len(state_b.on_exit), 1)
        self.loop.run_until_complete(model.to_B())
        self.assertTrue(model.exit_A_called)
        self.loop.run_until_complete(model.to_A())
        self.assertTrue(model.exit_B_called)
        with self.assertRaises(AttributeError):
            self.loop.run_until_complete(model.to_C())

    def test_pickle(self):
        import pickle

        states = ['A', 'B', 'C', 'D']
        # Define with list of dictionaries
        transitions = [
            {'trigger': 'walk', 'source': 'A', 'dest': 'B'},
            {'trigger': 'run', 'source': 'B', 'dest': 'C'},
            {'trigger': 'sprint', 'source': 'C', 'dest': 'D'}
        ]
        m = AsyncMachine(states=states, transitions=transitions, initial='A')
        self.loop.run_until_complete(m.walk())
        dump = pickle.dumps(m)
        self.assertIsNotNone(dump)
        m2 = pickle.loads(dump)
        self.assertEqual(m.state, m2.state)
        self.loop.run_until_complete(m2.run())

    def test_pickle_model(self):
        import pickle

        self.loop.run_until_complete(self.stuff.to_B())
        dump = pickle.dumps(self.stuff)
        self.assertIsNotNone(dump)
        model2 = pickle.loads(dump)
        self.assertEqual(self.stuff.state, model2.state)
        self.loop.run_until_complete(model2.to_F())

    def test_queued(self):
        states = ['A', 'B', 'C', 'D']
        # Define with list of dictionaries

        async def change_state(machine):
            self.assertEqual(machine.state, 'A')
            if machine.has_queue:
                await machine.run(machine=machine)
                self.assertEqual(machine.state, 'A')
            else:
                with self.assertRaises(MachineError):
                    await machine.run(machine=machine)

        transitions = [
            {'trigger': 'walk', 'source': 'A', 'dest': 'B', 'before': change_state},
            {'trigger': 'run', 'source': 'B', 'dest': 'C'},
            {'trigger': 'sprint', 'source': 'C', 'dest': 'D'}
        ]

        m = AsyncMachine(states=states, transitions=transitions, initial='A')
        self.loop.run_until_complete(m.walk(machine=m))
        self.assertEqual(m.state, 'B')
        m = AsyncMachine(states=states, transitions=transitions, initial='A', queued=True)
        self.loop.run_until_complete(m.walk(machine=m))
        self.assertEqual(m.state, 'C')

    def test_queued_errors(self):
        async def before_change(machine):
            if machine.has_queue:
                await machine.to_A(machine)
            machine._queued = False

        async def after_change(machine):
            await machine.to_C(machine)

        def failed_transition(machine):
            raise ValueError('Something was wrong')

        states = ['A', 'B', 'C']
        transitions = [{'trigger': 'do', 'source': '*', 'dest': 'C', 'before': failed_transition}]
        m = AsyncMachine(states=states, transitions=transitions, queued=True,
                         before_state_change=before_change, after_state_change=after_change)
        with self.assertRaises(MachineError):
            self.loop.run_until_complete(m.to_B(machine=m))

        with self.assertRaises(ValueError):
            self.loop.run_until_complete(m.do(machine=m))

    def test___getattr___and_identify_callback(self):
        m = AsyncMachine(AsyncStuff(), states=['A', 'B', 'C'], initial='A')
        m.add_transition('move', 'A', 'B')
        m.add_transition('move', 'B', 'C')

        callback = m.__getattr__('before_move')
        self.assertTrue(callable(callback))

        with self.assertRaises(AttributeError):
            m.__getattr__('before_no_such_transition')

        with self.assertRaises(AttributeError):
            m.__getattr__('before_no_such_transition')

        with self.assertRaises(AttributeError):
            m.__getattr__('__no_such_method__')

        with self.assertRaises(AttributeError):
            m.__getattr__('')

        type, target = m._identify_callback('on_exit_foobar')
        self.assertEqual(type, 'on_exit')
        self.assertEqual(target, 'foobar')

        type, target = m._identify_callback('on_exitfoobar')
        self.assertEqual(type, None)
        self.assertEqual(target, None)

        type, target = m._identify_callback('notacallback_foobar')
        self.assertEqual(type, None)
        self.assertEqual(target, None)

        type, target = m._identify_callback('totallyinvalid')
        self.assertEqual(type, None)
        self.assertEqual(target, None)

        type, target = m._identify_callback('before__foobar')
        self.assertEqual(type, 'before')
        self.assertEqual(target, '_foobar')

        type, target = m._identify_callback('before__this__user__likes__underscores___')
        self.assertEqual(type, 'before')
        self.assertEqual(target, '_this__user__likes__underscores___')

        type, target = m._identify_callback('before_stuff')
        self.assertEqual(type, 'before')
        self.assertEqual(target, 'stuff')

        type, target = m._identify_callback('before_trailing_underscore_')
        self.assertEqual(type, 'before')
        self.assertEqual(target, 'trailing_underscore_')

        type, target = m._identify_callback('before_')
        self.assertIs(type, None)
        self.assertIs(target, None)

        type, target = m._identify_callback('__')
        self.assertIs(type, None)
        self.assertIs(target, None)

        type, target = m._identify_callback('')
        self.assertIs(type, None)
        self.assertIs(target, None)

    def test_state_and_transition_with_underscore(self):
        m = AsyncMachine(AsyncStuff(), states=['_A_', '_B_', '_C_'], initial='_A_')
        m.add_transition('_move_', '_A_', '_B_', prepare='increase_level')
        m.add_transition('_after_', '_B_', '_C_', prepare='increase_level')
        m.add_transition('_on_exit_', '_C_', '_A_', prepare='increase_level', conditions='this_fails')

        self.loop.run_until_complete(m.model._move_())
        self.assertEqual(m.model.state, '_B_')
        self.assertEqual(m.model.level, 2)

        self.loop.run_until_complete(m.model._after_())
        self.assertEqual(m.model.state, '_C_')
        self.assertEqual(m.model.level, 3)

        # State does not advance, but increase_level still runs
        self.loop.run_until_complete(m.model._on_exit_())
        self.assertEqual(m.model.state, '_C_')
        self.assertEqual(m.model.level, 4)

    def test_callback_identification(self):
        m = AsyncMachine(AsyncStuff(), states=['A', 'B', 'C', 'D', 'E', 'F'], initial='A')
        m.add_transition('transition', 'A', 'B', before='increase_level')
        m.add_transition('after', 'B', 'C', before='increase_level')
        m.add_transition('on_exit_A', 'C', 'D', before='increase_level', conditions='this_fails')
        m.add_transition('check', 'C', 'E', before='increase_level')
        m.add_transition('prepare', 'E', 'F', before='increase_level')
        m.add_transition('before', 'F', 'A', before='increase_level')

        m.before_transition('increase_level')
        m.before_after('increase_level')
        m.before_on_exit_A('increase_level')
        m.after_check('increase_level')
        m.before_prepare('increase_level')
        m.before_before('increase_level')

        self.loop.run_until_complete(m.model.transition())
        self.assertEqual(m.model.state, 'B')
        self.assertEqual(m.model.level, 3)

        self.loop.run_until_complete(m.model.after())
        self.assertEqual(m.model.state, 'C')
        self.assertEqual(m.model.level, 5)

        self.loop.run_until_complete(m.model.on_exit_A())
        self.assertEqual(m.model.state, 'C')
        self.assertEqual(m.model.level, 5)

        self.loop.run_until_complete(m.model.check())
        self.assertEqual(m.model.state, 'E')
        self.assertEqual(m.model.level, 7)

        self.loop.run_until_complete(m.model.prepare())
        self.assertEqual(m.model.state, 'F')
        self.assertEqual(m.model.level, 9)

        self.loop.run_until_complete(m.model.before())
        self.assertEqual(m.model.state, 'A')
        self.assertEqual(m.model.level, 11)

        # An invalid transition shouldn't execute the callback
        with self.assertRaises(MachineError):
            self.loop.run_until_complete(m.model.on_exit_A())

    def test_process_trigger(self):
        m = AsyncMachine(states=['raw', 'processed'], initial='raw')
        m.add_transition('process', 'raw', 'processed')

        self.loop.run_until_complete(m.process())
        self.assertEqual(m.state, 'processed')

    def test_multiple_models(self):
        s1, s2 = AsyncStuff(), AsyncStuff()
        states = ['A', 'B', 'C']

        m = AsyncMachine(model=[s1, s2], states=states,
                         initial=states[0])
        self.assertEqual(len(m.models), 2)
        self.assertEqual(len(m.model), 2)
        m.add_transition('advance', 'A', 'B')
        self.loop.run_until_complete(s1.advance())
        self.assertEqual(s1.state, 'B')
        self.assertEqual(s2.state, 'A')
        m = AsyncMachine(model=s1, states=states,
                         initial=states[0])
        # for backwards compatibility model should return a model instance
        # rather than a list
        self.assertNotIsInstance(m.model, list)

    def test_dispatch(self):
        s1, s2 = AsyncStuff(), AsyncStuff()
        states = ['A', 'B', 'C']
        m = AsyncMachine(model=s1, states=states, ignore_invalid_triggers=True,
                         initial=states[0], transitions=[['go', 'A', 'B'], ['go', 'B', 'C']])
        m.add_model(s2, initial='B')
        self.loop.run_until_complete(m.dispatch('go'))
        self.assertEqual(s1.state, 'B')
        self.assertEqual(s2.state, 'C')

    def test_string_trigger(self):
        def return_value(value):
            return value

        class Model:
            def trigger(self, value):
                return value

        self.stuff.machine.add_transition('do', '*', 'C')
        self.loop.run_until_complete(self.stuff.trigger('do'))
        self.assertTrue(self.stuff.is_C())
        self.stuff.machine.add_transition('maybe', 'C', 'A', conditions=return_value)
        self.assertFalse(self.loop.run_until_complete(self.stuff.trigger('maybe', value=False)))
        self.assertTrue(self.loop.run_until_complete(self.stuff.trigger('maybe', value=True)))
        self.assertTrue(self.stuff.is_A())
        with self.assertRaises(AttributeError):
            self.loop.run_until_complete(self.stuff.trigger('not_available'))

        model = Model()
        m = AsyncMachine(model=model)
        self.assertEqual(model.trigger(5), 5)

    def test_get_triggers(self):
        states = ['A', 'B', 'C']
        transitions = [['a2b', 'A', 'B'],
                       ['a2c', 'A', 'C'],
                       ['c2b', 'C', 'B']]
        machine = AsyncMachine(states=states, transitions=transitions, initial='A', auto_transitions=False)
        self.assertEqual(len(machine.get_triggers('A')), 2)
        self.assertEqual(len(machine.get_triggers('B')), 0)
        self.assertEqual(len(machine.get_triggers('C')), 1)
        # self stuff machine should have to-transitions to every state
        self.assertEqual(len(self.stuff.machine.get_triggers('B')), len(self.stuff.machine.states))

    def test_repr(self):
        def a_condition(event_data):
            self.assertRegex(
                str(event_data.transition.conditions),
                r"\[<AsyncCondition\(<function TestTransitions.test_repr.<locals>"
                r".a_condition at [^>]+>\)@\d+>\]")
            return True

        # No transition has been assigned to EventData yet
        def check_prepare_repr(event_data):
            self.assertRegex(
                str(event_data),
                r"<EventData\('<AsyncState\('A'\)@\d+>', "
                r"None\)@\d+>")

        def check_before_repr(event_data):
            self.assertRegex(
                str(event_data),
                r"<EventData\('<AsyncState\('A'\)@\d+>', "
                r"<AsyncTransition\('A', 'B'\)@\d+>\)@\d+>")
            m.checked = True

        m = AsyncMachine(states=['A', 'B'],
                         prepare_event=check_prepare_repr,
                         before_state_change=check_before_repr, send_event=True,
                         initial='A')
        m.add_transition('do_strcheck', 'A', 'B', conditions=a_condition)

        self.assertTrue(self.loop.run_until_complete(m.do_strcheck()))
        self.assertIn('checked', vars(m))

    def test_machine_prepare(self):

        global_mock = MagicMock()
        local_mock = MagicMock()

        async def global_callback():
            global_mock()

        async def local_callback():
            local_mock()

        async def always_fails():
            return False

        transitions = [
            {'trigger': 'go', 'source': 'A', 'dest': 'B', 'conditions': always_fails, 'prepare': local_callback},
            {'trigger': 'go', 'source': 'A', 'dest': 'B', 'conditions': always_fails, 'prepare': local_callback},
            {'trigger': 'go', 'source': 'A', 'dest': 'B', 'conditions': always_fails, 'prepare': local_callback},
            {'trigger': 'go', 'source': 'A', 'dest': 'B', 'conditions': always_fails, 'prepare': local_callback},
            {'trigger': 'go', 'source': 'A', 'dest': 'B', 'prepare': local_callback},

        ]
        m = AsyncMachine(states=['A', 'B'], transitions=transitions,
                         prepare_event=global_callback, initial='A')

        self.loop.run_until_complete(m.go())
        self.assertEqual(global_mock.call_count, 1)
        self.assertEqual(local_mock.call_count, len(transitions))

    def test_machine_finalize(self):

        finalize_mock = MagicMock()

        def always_fails(event_data):
            return False

        def always_raises(event_data):
            raise Exception()

        transitions = [
            {'trigger': 'go', 'source': 'A', 'dest': 'B'},
            {'trigger': 'planA', 'source': 'B', 'dest': 'A', 'conditions': always_fails},
            {'trigger': 'planB', 'source': 'B', 'dest': 'A', 'conditions': always_raises}
        ]
        m = self.stuff.machine_cls(states=['A', 'B'], transitions=transitions,
                                   finalize_event=finalize_mock, initial='A', send_event=True)

        self.loop.run_until_complete(m.go())
        self.assertEqual(finalize_mock.call_count, 1)
        self.loop.run_until_complete(m.planA())

        event_data = finalize_mock.call_args[0][0]
        self.assertIsInstance(event_data, EventData)
        self.assertEqual(finalize_mock.call_count, 2)
        self.assertFalse(event_data.result)
        with self.assertRaises(Exception):
            self.loop.run_until_complete(m.planB())
        self.assertEqual(finalize_mock.call_count, 3)

    def test_machine_finalize_exception(self):

        exception = ZeroDivisionError()

        async def always_raises(event):
            raise exception

        async def finalize_callback(event):
            self.assertEqual(event.error, exception)

        m = self.stuff.machine_cls(states=['A', 'B'], send_event=True, initial='A',
                                   before_state_change=always_raises,
                                   finalize_event=finalize_callback)

        with self.assertRaises(ZeroDivisionError):
            self.loop.run_until_complete(m.to_B())

    def test_prep_ordered_arg(self):
        self.assertTrue(len(_prep_ordered_arg(3, None)) == 3)
        self.assertTrue(all(a is None for a in _prep_ordered_arg(3, None)))
        with self.assertRaises(ValueError):
            _prep_ordered_arg(3, [None, None])

    def test_ordered_transition_callback(self):
        class Model:
            def __init__(self):
                self.flag = False

            async def make_true(self):
                self.flag = True

        model = Model()
        states = ['beginning', 'middle', 'end']
        transits = [None, None, 'make_true']
        m = AsyncMachine(model, states, initial='beginning')
        m.add_ordered_transitions(before=transits)
        self.loop.run_until_complete(model.next_state())
        self.assertFalse(model.flag)
        self.loop.run_until_complete(model.next_state())
        self.loop.run_until_complete(model.next_state())
        self.assertTrue(model.flag)

    def test_ordered_transition_condition(self):
        class Model:
            def __init__(self):
                self.blocker = False

            async def check_blocker(self):
                return self.blocker

        model = Model()
        states = ['beginning', 'middle', 'end']
        m = AsyncMachine(model, states, initial='beginning')
        m.add_ordered_transitions(conditions=[None, None, 'check_blocker'])
        self.loop.run_until_complete(model.to_end())
        self.assertFalse(self.loop.run_until_complete(model.next_state()))
        model.blocker = True
        self.assertTrue(self.loop.run_until_complete(model.next_state()))

    def test_get_transitions(self):
        states = ['A', 'B', 'C', 'D']
        m = AsyncMachine('self', states, initial='a', auto_transitions=False)
        m.add_transition('go', ['A', 'B', 'C'], 'D')
        m.add_transition('run', 'A', 'D')
        self.assertEqual(
            {(t.source, t.dest) for t in m.get_transitions('go')},
            {('A', 'D'), ('B', 'D'), ('C', 'D')})
        self.assertEqual(
            [(t.source, t.dest)
             for t in m.get_transitions(source='A', dest='D')],
            [('A', 'D'), ('A', 'D')])

    def test_remove_transition(self):
        self.stuff.machine.add_transition('go', ['A', 'B', 'C'], 'D')
        self.stuff.machine.add_transition('walk', 'A', 'B')
        self.loop.run_until_complete(self.stuff.go())
        self.assertEqual(self.stuff.state, 'D')
        self.loop.run_until_complete(self.stuff.to_A())
        self.stuff.machine.remove_transition('go', source='A')
        with self.assertRaises(MachineError):
            self.loop.run_until_complete(self.stuff.go())
        self.stuff.machine.add_transition('go', 'A', 'D')
        self.loop.run_until_complete(self.stuff.walk())
        self.loop.run_until_complete(self.stuff.go())
        self.assertEqual(self.stuff.state, 'D')
        self.loop.run_until_complete(self.stuff.to_C())
        self.stuff.machine.remove_transition('go', dest='D')
        self.assertFalse(hasattr(self.stuff, 'go'))

    def test_reflexive_transition(self):
        self.stuff.machine.add_transition('reflex', ['A', 'B'], '=', after='increase_level')
        self.assertEqual(self.stuff.state, 'A')
        self.loop.run_until_complete(self.stuff.reflex())
        self.assertEqual(self.stuff.state, 'A')
        self.assertEqual(self.stuff.level, 2)
        self.loop.run_until_complete(self.stuff.to_B())
        self.assertEqual(self.stuff.state, 'B')
        self.loop.run_until_complete(self.stuff.reflex())
        self.assertEqual(self.stuff.state, 'B')
        self.assertEqual(self.stuff.level, 3)
        self.loop.run_until_complete(self.stuff.to_C())
        with self.assertRaises(MachineError):
            self.loop.run_until_complete(self.stuff.reflex())
        self.assertEqual(self.stuff.level, 3)

    def test_internal_transition(self):
        m = AsyncMachine(AsyncStuff(), states=['A', 'B'], initial='A')
        m.add_transition('move', 'A', None, prepare='increase_level')
        self.loop.run_until_complete(m.model.move())
        self.assertEqual(m.model.state, 'A')
        self.assertEqual(m.model.level, 2)


class TestWarnings(TestCase):

    def test_warning(self):
        # does not work with python 3.3. However, the warning is shown when Machine is initialized manually.
        if (3, 3) <= sys.version_info < (3, 4):
            return

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings(action='default', message=r"Starting from transitions version 0\.6\.0 .*")
            m = AsyncMachine(None)
            m = AsyncMachine(add_self=False)
            self.assertEqual(len(w), 1)
            for warn in w:
                self.assertEqual(warn.category, DeprecationWarning)
