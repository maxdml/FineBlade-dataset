import numpy as np
import sys
import os
import json
import time
from ..logger import *
import pandas as pd
from CPE_analytic.utils import *
from sqlalchemy import create_engine

import re

class DbApi:
    def __init__(self, dfg_file=None,
                 db_ip=None, db_port=None, db_user=None, db_pwd=None, db_name=None):
        self.utils = utils()

        if dfg_file is not None:
            self.json_f = dfg_file
            fp = open(self.json_f, 'r')
            self.json = json.load(fp)
            fp.close()

            log_info("Setting database to ip: %s, port %d" % (self.json['db_ip'], self.json['db_port']))
            self.engine = create_engine('mysql://{db_user}:{db_pwd}@{db_ip}:{db_port}/{db_name}'.format(**self.json))
        else:
            db_config = dict(
                    db_ip=db_ip,
                    db_port=db_port,
                    db_user=db_user,
                    db_pwd=db_pwd,
                    db_name=db_name
            )
            log_info("Setting database to ip {db_ip}, port {db_port}".format(**db_config))
            self.engine = create_engine('mysql://{db_user}:{db_pwd}@{db_ip}:{db_port}/{db_name}'.format(**db_config))

    def select_query(self, table, columns=None, **kwargs):
        if columns is None:
            selection = '*'
        else:
            selection = ', '.join(columns)

        query = 'SELECT {} FROM {}'.format(selection, table)

        if len(kwargs) > 0:
            for i, (k, v) in enumerate(filter(lambda x: x[1] is not None, kwargs.items())):
                if i == 0:
                    query += ' WHERE '
                else:
                    query += ' AND '
                if isinstance(v, basestring) or isinstance(v, int):
                    v = (v,)

                if isinstance(next(vi for vi in v), basestring):
                    query += '{} in ({})'.format(k, ','.join("'{}'".format(vv) for vv in v))
                else:
                    query += '{} in ({})'.format(k, ','.join(str(vv) for vv in v))
        return query

    def select_from(self, table, columns=None, **kwargs):
        query = self.select_query(table, columns, **kwargs)
        return pd.read_sql_query(query, self.engine)

    def get_stats(self, name=None):
        return self.select_from('Statistics', name=name)

    def get_msus(self, msu_pk=None, msu_id=None, msu_type_id=None, thread_pk=None,
                 msu_type_name=None, thread_id=None, runtime_id=None):
        threads = self.get_threads(thread_id=thread_id, runtime_id=runtime_id)
        msu_types = self.get_msu_types(msu_type_name)
        msus = self.select_from('Msus',
                                pk=msu_pk, msu_id=msu_id,
                                msu_type_id=msu_type_id, thread_pk=thread_pk)

        return msus.join(threads.set_index('pk'), on='thread_pk', how='inner') \
                   .join(msu_types.set_index('id'), on='msu_type_id', how='inner')

    def get_msu_types(self, name=None):
        return self.select_from('MsuTypes', name=name)

    def get_runtimes(self, runtime_id=None):
        return self.select_from('Runtimes', id=runtime_id)

    def get_threads(self, thread_id=None, runtime_id=None):
        return self.select_from('Threads', thread_id=thread_id, runtime_id=runtime_id)

    def get_bin_percentiles(self):
        return pd.read_sql_query('SELECT percentile FROM (SELECT percentile FROM Bins LIMIT 50) bins GROUP BY percentile', self.engine)

    def get_timeseries(self, msu_pk=None, msu_id=None, msus=None,
                       thread_pk=None, thread_id=None, threads=None,
                       runtime_id=None, stat_name=None, stat_id=None):

        if msu_id is not None:
            msus = self.select_from('Msus', ('pk', 'msu_id', 'msu_type_id'), msu_id=msu_id)
            msu_pk = msus.pk
        elif msus is not None:
            msu_pk = msus.pk.values
        else:
            msus = self.get_msus()

        if thread_id is not None:
            threads = self.select_from('Threads', ('pk',), thread_id=thread_id)
            thread_pk = threads.pk
        elif threads is not None:
            thread_pk = threads.pk.values

        if stat_id is not None:
            stats = self.select_from('Statistics', id=stat_id)
        elif stat_name is not None:
            stats = self.select_from('Statistics', name=stat_name)
            stat_id = stats.id
        else:
            stats = self.select_from('Statistics')

        all_none = msu_pk is None and thread_pk is None and runtime_id is None
        timeseries = self.select_from('Timeseries',
                                      msu_pk=msu_pk, runtime_id=runtime_id, thread_pk=thread_pk,
                                      statistic_id = stat_id).rename(columns={'pk':'ts_pk'})

        merged = timeseries.merge(stats, left_on='statistic_id', right_on='id')

        columns = ['ts_pk', 'name', 'monotonic']

        if (merged['msu_pk'] > 0).any():
            columns.append('msu_pk')
            columns.append('msu_type_id')
            mpks = msus[msus.pk.isin(merged.msu_pk.unique())][['msu_type_id', 'pk']]\
                   .rename(columns={'pk': 'msu_pk'})
            merged = merged.merge(mpks, how='outer', on='msu_pk')
        if (merged['thread_pk'] > 0).any():
            columns.append('thread_pk')
        if (merged['runtime_id'] > 0).any():
            columns.append('runtime_id')

        return merged[columns].rename(columns={'ts_pk': 'pk'})

    def points_query(self, timeseries_pk, start=0, end=None, fields=('ts', 'size', 'pk')):

        if end is not None:
            points_where = 'ts BETWEEN {} AND {}'.format(start, end)
        else:
            points_where = 'ts > {}'.format(start)

        query = self.select_query('Points', fields, timeseries_pk=timeseries_pk)
        query += ' AND ' + points_where

        return query

    def bins_query(self, timeseries_pk, percentile=0, start=0, end=None):

        points = self.points_query(timeseries_pk, start, end, ('ts', 'size', 'pk', 'timeseries_pk'))

        try:
            where_clause = 'bn.percentile in ({})'.format(','.join([str(x) for x in percentile]))
        except Exception:
            where_clause = 'bn.percentile = {}'.format(percentile)

        query = ('SELECT FLOOR(pt.ts * 10) / 10 as ts, pt.size, bn.value, bn.percentile, pt.timeseries_pk FROM ({}) pt ' +
                 'INNER JOIN Bins bn ON bn.points_pk = pt.pk WHERE {}')\
                         .format(points, where_clause)

        return query

    def get_bins(self, timeseries_pk, percentile=0, start=0, end=None):
        query = self.bins_query(timeseries_pk, percentile, start, end)
        df = pd.read_sql_query(query, self.engine)
        return df


    def get_monotonic_df(self, timeseries_pk, start, end):
        fields = 'FLOOR(ts * 10) / 10 as ts', 'size as value', 'timeseries_pk'
        query = self.points_query(timeseries_pk, start, end, fields)
        df = pd.read_sql_query(query, self.engine)
        return df

    def get_events(self, reset_time=True):
        events = self.select_from('Events')

        if reset_time:
            events = events.rename(columns={'ts': 'time'})
            events.time -= self.get_start_time()

        return events

    @staticmethod
    def concat_stat_timeseries(df1, df2, join_how):
        index_df1 = df1.set_index(['ts', 'percentile'])
        if not index_df1.index.is_unique:
            index_df1 = df1.groupby(['ts', 'percentile']).mean()

        index_df2 = df2.set_index(['ts', 'percentile'])
        if not index_df2.index.is_unique:
            index_df2 = df2.groupby(['ts', 'percentile']).mean()

        return pd.concat([index_df1, index_df2],
                         axis=1, join=join_how).reset_index()

    @staticmethod
    def label_timeseries(ts_df, events_df):
        ts_df['traffic'] = ''
        started = []
        for _, event in events_df.iterrows():
            if event.status == 'start':
                started.append(event['name'])
            else:
                started.pop()
            if len(started) > 0:
                ts_df.loc[ts_df.ts > event.time, 'traffic'] = started[-1]
            else:
                ts_df.loc[ts_df.ts > event.time, 'traffic'] = ''
        return ts_df

    @staticmethod
    def flatten_timeseries(df, cols):
        piv = df.pivot_table(index='ts', values='value', columns = cols)
        if len(cols) == 1:
            return piv.reset_index()

        levels = piv.columns.levels
        labels = piv.columns.labels
        names = piv.columns.names

        lvls = [np.array([n +'_'+ str(x) for x in l]) for n, l in zip(piv.columns.names, levels)]

        cols = []
        for lvl, label in zip(lvls, labels):
            cols.append(lvl[label])

        newcols = []
        for c in zip(*cols):
            newcols.append('-'.join(c))

        piv.columns = newcols
        return piv.reset_index()


    def get_multi_stat_timeseries(self, percentile=0, start=0, end=None,
                                  reset_time=True, round_to=None, dropna=False, **kwargs):

        timeseries = self.get_timeseries(**kwargs)

        columns = timeseries.columns
        columns = columns[columns != 'pk']
        columns = columns[columns != 'monotonic']

        full_df = pd.DataFrame(columns=columns)

        if reset_time:
            start += self.get_start_time()
            if end is not None:
                end += self.get_start_time()

        for _, ts in timeseries.groupby('monotonic'):
            log_debug("Retrieving {} timeseries".format(len(ts)))
            dbg_start = time.time()
            if not ts['monotonic'].any():
                df = self.get_bins(ts.pk, percentile, start, end)
                if df is None or df.empty:
                    continue
            else:
                df = self.get_monotonic_df(ts.pk, start, end)
            log_debug("{} timeseries got in {} seconds".format(len(ts), time.time() - dbg_start))

            df = df.merge(ts, left_on='timeseries_pk', right_on='pk')

            if ts.monotonic.any():
                df['percentile'] = 0

            if round_to is not None:
                df.ts = np.ceil(df.ts)

            full_df = full_df.append(df)

        if full_df is None or full_df.empty:
            print('No stats found!')
            return None

        if dropna:
            full_df = full_df.dropna(1, 'all').dropna(0, 'any')

        if reset_time:
            full_df.ts -= self.get_start_time()

        return full_df

    def get_start_time(self):
        return pd.read_sql_query('SELECT FLOOR(MIN(ts)) as start FROM Points', self.engine)['start'][0]

    def get_end_time(self):
        return pd.read_sql_query('SELECT FLOOR(MAX(ts)) as end FROM Points', self.engine)['end'][0]
