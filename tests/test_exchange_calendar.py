#
# Copyright 2016 Quantopian, Inc.
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

from os.path import (
    abspath,
    dirname,
    join,
)
from unittest import TestCase
from collections import namedtuple

import numpy as np
import pandas as pd
import pytz
from pandas import (
    read_csv,
    datetime,
    Timestamp,
)
from pandas.util.testing import assert_frame_equal

from zipline.errors import (
    CalendarNameCollision,
    InvalidCalendarName,
)
from zipline.utils.calendars.exchange_calendar_nyse import NYSEExchangeCalendar
from zipline.utils.calendars.exchange_calendar import(
    register_calendar,
    deregister_calendar,
    get_calendar,
    clear_calendars,
)


class CalendarRegistrationTestCase(TestCase):

    def setUp(self):
        self.dummy_cal_type = namedtuple('DummyCal', ('name'))

    def tearDown(self):
        clear_calendars()

    def test_register_calendar(self):
        # Build a fake calendar
        dummy_cal = self.dummy_cal_type('DMY')

        # Try to register and retrieve the calendar
        register_calendar(dummy_cal)
        retr_cal = get_calendar('DMY')
        self.assertEqual(dummy_cal, retr_cal)

        # Try to register again, expecting a name collision
        with self.assertRaises(CalendarNameCollision):
            register_calendar(dummy_cal)

        # Deregister the calendar and ensure that it is removed
        deregister_calendar('DMY')
        with self.assertRaises(InvalidCalendarName):
            get_calendar('DMY')

    def test_force_registration(self):
        dummy_nyse = self.dummy_cal_type('NYSE')

        # Get the actual NYSE calendar
        real_nyse = get_calendar('NYSE')

        # Force a registration of the dummy NYSE
        register_calendar(dummy_nyse, force=True)

        # Ensure that the dummy overwrote the real calendar
        retr_cal = get_calendar('NYSE')
        self.assertNotEqual(real_nyse, retr_cal)


class ExchangeCalendarTestBase(object):

    # Override in subclasses.
    answer_key_filename = None
    calendar_class = None

    @staticmethod
    def load_answer_key(filename):
        """
        Load a CSV from tests/resources/calendars/{filename}.csv
        """
        fullpath = join(
            dirname(abspath(__file__)),
            'resources',
            'calendars',
            filename + '.csv',
        )
        return read_csv(
            fullpath,
            index_col=0,
            # NOTE: Merely passing parse_dates=True doesn't cause pandas to set
            # the dtype correctly, and passing all reasonable inputs to the
            # dtype kwarg cause read_csv to barf.
            parse_dates=[0, 1, 2],
        ).tz_localize('UTC')

    @classmethod
    def setupClass(cls):
        cls.answers = cls.load_answer_key(cls.answer_key_filename)
        cls.start_date = cls.answers.index[0]
        cls.end_date = cls.answers.index[-1]
        cls.calendar = cls.calendar_class(cls.start_date, cls.end_date)

        cls.one_minute = pd.Timedelta(minutes=1)
        cls.one_hour = pd.Timedelta(hours=1)

    def test_calculated_against_csv(self):
        assert_frame_equal(self.calendar.schedule, self.answers)

    def test_is_open_on_minute(self):
        one_minute = pd.Timedelta(minutes=1)

        for market_minute in self.answers.market_open:
            market_minute_utc = market_minute.tz_localize('UTC')
            # The exchange should be classified as open on its first minute
            self.assertTrue(self.calendar.is_open_on_minute(market_minute_utc))

            # Decrement minute by one, to minute where the market was not open
            pre_market = market_minute_utc - one_minute
            self.assertFalse(self.calendar.is_open_on_minute(pre_market))

        for market_minute in self.answers.market_close:
            close_minute_utc = market_minute.tz_localize('UTC')
            # should be open on its last minute
            self.assertTrue(self.calendar.is_open_on_minute(close_minute_utc))

            # increment minute by one minute, should be closed
            post_market = close_minute_utc + one_minute
            self.assertFalse(self.calendar.is_open_on_minute(post_market))

    def test_open_and_close(self):
        for index, row in self.answers.iterrows():
            o_and_c = self.calendar.open_and_close(index)
            self.assertEqual(o_and_c[0],
                             row['market_open'].tz_localize('UTC'))
            self.assertEqual(o_and_c[1],
                             row['market_close'].tz_localize('UTC'))

    def _verify_minute(self, calendar, minute,
                      next_open_answer, prev_open_answer,
                      next_close_answer, prev_close_answer):
        self.assertEqual(
            calendar.next_open(minute),
            next_open_answer
        )

        self.assertEqual(
            self.calendar.previous_open(minute),
            prev_open_answer
        )

        self.assertEqual(
            self.calendar.next_close(minute),
            next_close_answer
        )

        self.assertEqual(
            self.calendar.previous_close(minute),
            prev_close_answer
        )

    def test_next_prev_open_close(self):
        # for each session, check:
        # - the minute before the open
        # - the first minute of the session
        # - the second minute of the session
        # - the minute before the close
        # - the last minute of the session
        # - the first minute after the close
        answers_to_use = self.answers[1:-2]

        for idx, info in enumerate(answers_to_use.iterrows()):
            open_minute = info[1].market_open
            close_minute = info[1].market_close

            minute_before_open = open_minute - self.one_minute

            # answers_to_use starts at the second element of self.answers,
            # so self.answers.iloc[idx] is one element before, and
            # self.answers.iloc[idx + 2] is one element after the current
            # element
            previous_open = self.answers.iloc[idx].market_open
            next_open = self.answers.iloc[idx + 2].market_open
            previous_close = self.answers.iloc[idx].market_close
            next_close = self.answers.iloc[idx + 2].market_close

            # minute before open
            self._verify_minute(self.calendar, minute_before_open,
                open_minute, previous_open,
                close_minute, previous_close
            )

            # open minute
            self._verify_minute(self.calendar, open_minute,
                next_open, previous_open,
                close_minute, previous_close
            )

            # second minute of session
            self._verify_minute(self.calendar, open_minute + self.one_minute,
                               next_open, open_minute,
                               close_minute, previous_close)

            # minute before the close
            self._verify_minute(self.calendar, close_minute - self.one_minute,
                               next_open, open_minute,
                               close_minute, previous_close)

            # the close
            self._verify_minute(self.calendar, close_minute,
                               next_open, open_minute,
                               next_close, previous_close)

            # minute after the close
            self._verify_minute(self.calendar, close_minute + self.one_minute,
                               next_open, open_minute,
                               next_close, close_minute)

    def test_next_prev_exchange_minute(self):
        all_minutes = self.calendar.all_trading_minutes

        # test 200,000 minutes because it takes too long to do the rest.
        # 200k minutes = ~2 years, which provides pretty good coverage.
        for idx, minute in enumerate(all_minutes[1:200000]):
            self.assertEqual(
                all_minutes[idx + 2],
                self.calendar.next_exchange_minute(minute)
            )

            self.assertEqual(
                all_minutes[idx],
                self.calendar.previous_exchange_minute(minute)
            )

        # test a couple of non-market minutes
        for open_minute in self.answers.market_open[1:]:
            hour_before_open = open_minute - self.one_hour
            self.assertEqual(
                open_minute.tz_localize("UTC"),
                self.calendar.next_exchange_minute(hour_before_open)
            )

        for close_minute in self.answers.market_close[1:]:
            hour_after_close = close_minute + self.one_hour
            self.assertEqual(
                close_minute.tz_localize("UTC"),
                self.calendar.previous_exchange_minute(hour_after_close)
            )

    def test_session_date(self):
        for idx, info in enumerate(self.answers[1:-2].iterrows()):
            session_label = info[1].name
            open_minute = info[1].market_open
            close_minute = info[1].market_close
            hour_into_session = open_minute + self.one_hour

            minute_before_session = open_minute - self.one_minute
            minute_after_session = close_minute + self.one_minute

            next_session_label = self.answers.iloc[idx + 2].name
            previous_session_label = self.answers.iloc[idx].name

            # verify that minutes inside a session resolve correctly
            minutes_that_resolve_to_this_session = [
                self.calendar.session_date(open_minute),
                self.calendar.session_date(open_minute, direction="next"),
                self.calendar.session_date(open_minute, direction="previous"),
                self.calendar.session_date(open_minute, direction="none"),
                self.calendar.session_date(hour_into_session),
                self.calendar.session_date(hour_into_session,
                                           direction="next"),
                self.calendar.session_date(hour_into_session,
                                           direction="previous"),
                self.calendar.session_date(hour_into_session,
                                           direction="none"),
                self.calendar.session_date(close_minute),
                self.calendar.session_date(close_minute, direction="next"),
                self.calendar.session_date(close_minute, direction="previous"),
                self.calendar.session_date(close_minute, direction="none"),
                self.calendar.session_date(minute_before_session),
                self.calendar.session_date(minute_before_session,
                                           direction="next"),
                self.calendar.session_date(minute_after_session,
                                           direction="previous"),
                session_label
            ]

            self.assertTrue(all(x == minutes_that_resolve_to_this_session[0]
                                for x in minutes_that_resolve_to_this_session))

            minutes_that_resolve_to_next_session = [
                self.calendar.session_date(minute_after_session),
                self.calendar.session_date(minute_after_session,
                                           direction="next"),
                next_session_label
            ]

            self.assertTrue(all(x == minutes_that_resolve_to_next_session[0]
                                for x in minutes_that_resolve_to_next_session))

            self.assertEqual(
                self.calendar.session_date(minute_before_session,
                                           direction="previous"),
                previous_session_label
            )

            # make sure that exceptions are raised at the right time
            with self.assertRaises(ValueError):
                self.calendar.session_date(open_minute, "asdf")

            with self.assertRaises(ValueError):
                self.calendar.session_date(minute_before_session,
                                           direction="none")

    def test_next_prev_session_label(self):
        session_labels = self.answers.index[1:-2]
        max_idx = len(session_labels) - 1

        # the very first session
        first_session_label = self.answers.index[0]
        with self.assertRaises(ValueError):
            self.calendar.previous_session_date(first_session_label)

        # all the sessions in the middle
        for idx, session in enumerate(session_labels):
            if idx < max_idx:
                self.assertEqual(
                    self.calendar.next_session_label(session),
                    session_labels[idx + 1]
                )

            if idx > 0:
                self.assertEqual(
                    self.calendar.previous_session_label(session),
                    session_labels[idx - 1]
                )

        # the very last session
        last_session_label = self.answers.index[-1]
        with self.assertRaises(ValueError):
            self.calendar.next_session_date(last_session_label)

    @staticmethod
    def _find_full_session(calendar):
        for session_label in calendar.schedule.index:
            if session_label not in calendar.early_closes:
                return session_label

        return None

    def test_minutes_for_session(self):
        # full session
        # find a session that isn't an early close.  start from the first
        # session, should be quick.
        full_session_label = self._find_full_session(self.calendar)
        if full_session_label is None:
            raise ValueError("Cannot find a full session to test!")

        minutes = self.calendar.minutes_for_session(full_session_label)
        _open, _close = self.calendar.open_and_close(full_session_label)

        np.testing.assert_array_equal(
            minutes,
            pd.date_range(start=_open, end=_close, freq="min")
        )

        # early close session
        early_close_session_label = self.calendar.early_closes[0]
        minutes_for_early_close = \
            self.calendar.minutes_for_session(early_close_session_label)
        _open, _close = self.calendar.open_and_close(early_close_session_label)

        np.testing.assert_array_equal(
            minutes_for_early_close,
            pd.date_range(start=_open, end=_close, freq="min")
        )

    def test_exchange_sessions_in_range(self):
        # pick two sessions
        session_count = len(self.calendar.schedule.index)

        first_idx = session_count / 3
        second_idx = 2 * first_idx

        first_session = self.calendar.schedule.index[first_idx]
        second_session = self.calendar.schedule.index[second_idx]

        answer_key = \
            self.calendar.schedule.index[first_idx:second_idx + 1]

        np.testing.assert_array_equal(
            answer_key,
            self.calendar.exchange_sessions_in_range(first_session,
                                                     second_session)
        )

    def test_exchange_minutes_in_range(self):
        # choose a 3-session period that surrounds a shortened session
        shortened_session = self.calendar.early_closes[0]
        shortened_session_idx = \
            self.calendar.schedule.index.get_loc(shortened_session)

        session_before = self.calendar.schedule.index[
            shortened_session_idx - 1
        ]
        session_after = self.calendar.schedule.index[shortened_session_idx + 1]

        first_open, first_close = self.calendar.open_and_close(session_before)
        minute_before_first_open = first_open - self.one_minute

        middle_open, middle_close = \
            self.calendar.open_and_close(shortened_session)

        last_open, last_close = self.calendar.open_and_close(session_after)
        minute_after_last_close = last_close + self.one_minute

        # get all the minutes between first_open and last_close
        minutes1 = self.calendar.exchange_minutes_in_range(
            first_open,
            last_close
        )
        minutes2 = self.calendar.exchange_minutes_in_range(
            minute_before_first_open,
            minute_after_last_close
        )

        np.testing.assert_array_equal(minutes1, minutes2)

        # manually construct the minutes
        all_minutes = np.concatenate([
            pd.date_range(
                start=first_open,
                end=first_close,
                freq="min"
            ),
            pd.date_range(
                start=middle_open,
                end=middle_close,
                freq="min"
            ),
            pd.date_range(
                start=last_open,
                end=last_close,
                freq="min"
            )
        ])

        np.testing.assert_array_equal(all_minutes, minutes1)

    def test_open_and_close(self):
        for index, row in self.answers.iterrows():
            session_label = row.name
            open_answer = row.market_open.tz_localize("UTC")
            close_answer = row.market_close.tz_localize("UTC")

            found_open, found_close = \
                self.calendar.open_and_close(session_label)

            self.assertEqual(open_answer, found_open)
            self.assertEqual(close_answer, found_close)


class NYSECalendarTestCase(ExchangeCalendarTestBase, TestCase):

    answer_key_filename = 'nyse'
    calendar_class = NYSEExchangeCalendar

    def test_newyears(self):
        """
        Check whether the ExchangeCalendar contains certain dates.
        """
        #     January 2012
        # Su Mo Tu We Th Fr Sa
        #  1  2  3  4  5  6  7
        #  8  9 10 11 12 13 14
        # 15 16 17 18 19 20 21
        # 22 23 24 25 26 27 28
        # 29 30 31

        start_dt = Timestamp('1/1/12', tz='UTC')
        end_dt = Timestamp('12/31/13', tz='UTC')
        sessions = self.calendar.exchange_sessions_in_range(start_dt, end_dt)

        day_after_new_years_sunday = datetime(2012, 1, 2, tzinfo=pytz.utc)

        self.assertNotIn(day_after_new_years_sunday, sessions,
                         """
 If NYE falls on a weekend, {0} the Monday after is a holiday.
 """.strip().format(day_after_new_years_sunday)
        )

        first_trading_day_after_new_years_sunday = datetime(
            2012, 1, 3, tzinfo=pytz.utc)

        self.assertIn(first_trading_day_after_new_years_sunday, sessions,
                      """
 If NYE falls on a weekend, {0} the Tuesday after is the first trading day.
 """.strip().format(first_trading_day_after_new_years_sunday)
        )

        #     January 2013
        # Su Mo Tu We Th Fr Sa
        #        1  2  3  4  5
        #  6  7  8  9 10 11 12
        # 13 14 15 16 17 18 19
        # 20 21 22 23 24 25 26
        # 27 28 29 30 31

        new_years_day = datetime(2013, 1, 1, tzinfo=pytz.utc)

        self.assertNotIn(new_years_day, sessions,
                         """
 If NYE falls during the week, e.g. {0}, it is a holiday.
 """.strip().format(new_years_day)
        )

        first_trading_day_after_new_years = datetime(
            2013, 1, 2, tzinfo=pytz.utc)

        self.assertIn(first_trading_day_after_new_years, sessions,
                      """
 If the day after NYE falls during the week, {0} \
 is the first trading day.
 """.strip().format(first_trading_day_after_new_years)
        )

    def test_thanksgiving(self):
        """
        Check ExchangeCalendar Thanksgiving dates.
        """
        #     November 2005
        # Su Mo Tu We Th Fr Sa
        #        1  2  3  4  5
        #  6  7  8  9 10 11 12
        # 13 14 15 16 17 18 19
        # 20 21 22 23 24 25 26
        # 27 28 29 30

        start_dt = Timestamp('1/1/05', tz='UTC')
        end_dt = Timestamp('12/31/12', tz='UTC')
        trading_days = self.calendar.trading_days(start=start_dt,
                                                  end=end_dt)

        thanksgiving_with_four_weeks = datetime(
            2005, 11, 24, tzinfo=pytz.utc)

        self.assertNotIn(thanksgiving_with_four_weeks,
                         trading_days.index,
                         """
 If Nov has 4 Thursdays, {0} Thanksgiving is the last Thursady.
 """.strip().format(thanksgiving_with_four_weeks)
        )

        #     November 2006
        # Su Mo Tu We Th Fr Sa
        #           1  2  3  4
        #  5  6  7  8  9 10 11
        # 12 13 14 15 16 17 18
        # 19 20 21 22 23 24 25
        # 26 27 28 29 30
        thanksgiving_with_five_weeks = datetime(
            2006, 11, 23, tzinfo=pytz.utc)

        self.assertNotIn(thanksgiving_with_five_weeks,
                         trading_days.index,
                         """
 If Nov has 5 Thursdays, {0} Thanksgiving is not the last week.
 """.strip().format(thanksgiving_with_five_weeks)
        )

        first_trading_day_after_new_years_sunday = datetime(
            2012, 1, 3, tzinfo=pytz.utc)

        self.assertIn(first_trading_day_after_new_years_sunday,
                      trading_days.index,
                      """
 If NYE falls on a weekend, {0} the Tuesday after is the first trading day.
 """.strip().format(first_trading_day_after_new_years_sunday)
        )

    def test_day_after_thanksgiving(self):
        #    November 2012
        # Su Mo Tu We Th Fr Sa
        #              1  2  3
        #  4  5  6  7  8  9 10
        # 11 12 13 14 15 16 17
        # 18 19 20 21 22 23 24
        # 25 26 27 28 29 30
        fourth_friday_open = Timestamp('11/23/2012 11:00AM', tz='EST')
        fourth_friday = Timestamp('11/23/2012 3:00PM', tz='EST')
        self.assertTrue(self.calendar.is_open_on_minute(fourth_friday_open))
        self.assertFalse(self.calendar.is_open_on_minute(fourth_friday))

        #    November 2013
        # Su Mo Tu We Th Fr Sa
        #                 1  2
        #  3  4  5  6  7  8  9
        # 10 11 12 13 14 15 16
        # 17 18 19 20 21 22 23
        # 24 25 26 27 28 29 30
        fifth_friday_open = Timestamp('11/29/2013 11:00AM', tz='EST')
        fifth_friday = Timestamp('11/29/2013 3:00PM', tz='EST')
        self.assertTrue(self.calendar.is_open_on_minute(fifth_friday_open))
        self.assertFalse(self.calendar.is_open_on_minute(fifth_friday))

    def test_early_close_independence_day_thursday(self):
        """
        Until 2013, the market closed early the Friday after an
        Independence Day on Thursday.  Since then, the early close is on
        Wednesday.
        """
        #      July 2002
        # Su Mo Tu We Th Fr Sa
        #     1  2  3  4  5  6
        #  7  8  9 10 11 12 13
        # 14 15 16 17 18 19 20
        # 21 22 23 24 25 26 27
        # 28 29 30 31
        wednesday_before = Timestamp('7/3/2002 3:00PM', tz='EST')
        friday_after_open = Timestamp('7/5/2002 11:00AM', tz='EST')
        friday_after = Timestamp('7/5/2002 3:00PM', tz='EST')
        self.assertTrue(self.calendar.is_open_on_minute(wednesday_before))
        self.assertTrue(self.calendar.is_open_on_minute(friday_after_open))
        self.assertFalse(self.calendar.is_open_on_minute(friday_after))

        #      July 2013
        # Su Mo Tu We Th Fr Sa
        #     1  2  3  4  5  6
        #  7  8  9 10 11 12 13
        # 14 15 16 17 18 19 20
        # 21 22 23 24 25 26 27
        # 28 29 30 31
        wednesday_before = Timestamp('7/3/2013 3:00PM', tz='EST')
        friday_after_open = Timestamp('7/5/2013 11:00AM', tz='EST')
        friday_after = Timestamp('7/5/2013 3:00PM', tz='EST')
        self.assertFalse(self.calendar.is_open_on_minute(wednesday_before))
        self.assertTrue(self.calendar.is_open_on_minute(friday_after_open))
        self.assertTrue(self.calendar.is_open_on_minute(friday_after))
