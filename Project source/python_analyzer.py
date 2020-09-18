#!/usr/bin/env python

from os import listdir, chdir, path
from pandas import read_csv, DataFrame, concat
from collections import OrderedDict
from random import randint, sample, seed
import matplotlib.pyplot as plt
import numpy as np
import csv

# 66 - testing seed
# 102 - training seed
seed(102)
num_sample_machines = 1000
machine_index = OrderedDict()

task_events_csv_colnames = ['time', 'missing', 'job_id', 'task_idx', 'machine_id', 'event_type', 'user', 'sched_cls',
                            'priority', 'cpu_requested', 'mem_requested', 'disk', 'restriction']
machine_events_csv_colnames = ['time', 'machine_id', 'event_type', 'platform_id', 'cpu', 'memory']
task_usage_csv_colnames = ['starttime', 'endtime', 'job_id', 'task_idx', 'machine_id', 'cpu_usage', 'mem_usage',
                           'assigned_mem', 'unmapped_cache_usage', 'page_cache_usage', 'max_mem_usage',
                           'disk_io_time', 'max_disk_space', 'max_cpu_usage', 'max_disk_io_time',
                           'cpi', 'mai', 'sampling_rate', 'agg_type']

task_usage_cols = [0, 1, 2, 3, 4, 5, 6]


def analyse_machine_usage(start, end, moments, machines, segment):
    sample_moments_iterator = iter(moments)
    current_sample_moment = next(sample_moments_iterator)
    moment_ind = 0

    res_dict = OrderedDict()
    for m in machines:
        res_dict[m] = OrderedDict([])
        mach_dict = res_dict[m]
        for s in moments:
            mach_dict[s] = OrderedDict()
            mach_dict[s]['cpu_usage'] = 0
            mach_dict[s]['mem_usage'] = 0

    for fn in sorted(listdir('task_usage'))[segment:]:
        print(fn)
        fp = path.join('task_usage', fn)
        trace_df = read_csv(fp, header=None, index_col=False, compression='gzip',
                            names=task_usage_csv_colnames, usecols=task_usage_cols)

        if start <= max(trace_df['starttime']):
            for index, event in trace_df.iterrows():
                if current_sample_moment is None:
                    break

                if event['machine_id'] in machines:
                    mach_dict = res_dict[event['machine_id']]

                    if event['starttime'] <= current_sample_moment <= event['endtime']:
                        mach_dict[current_sample_moment]['cpu_usage'] += event['cpu_usage']
                        mach_dict[current_sample_moment]['mem_usage'] += event['mem_usage']

                        # Check next sample moments
                        tmp = moment_ind + 1
                        while tmp < len(moments) and event['starttime'] <= moments[tmp] <= event['endtime']:
                            mach_dict[moments[tmp]]['cpu_usage'] += event['cpu_usage']
                            mach_dict[moments[tmp]]['mem_usage'] += event['mem_usage']
                            tmp += 1

                if event['starttime'] > current_sample_moment:
                    try:
                        current_sample_moment = next(sample_moments_iterator)
                        moment_ind += 1
                    except StopIteration:
                        current_sample_moment = None
        if max(trace_df['starttime']) > end or current_sample_moment is None:
            break

    return res_dict


def cpu_usage_plt(res_dict, start, end):
    count = 0
    for m_id in res_dict.keys():
        if count % 4 == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.xlabel('Time (h)')
            plt.ylabel('CPU rate usage (%)')
            plt.xlim((start - start) / 3600000000, (end - start) / 3600000000)

        m = res_dict[m_id]
        cpu_available = machine_index[m_id]['cpu_available']

        x_time = []
        y_cpu = []
        y_cpu_percentage = []
        for time in m.keys():
            x_time.append((time - start) / 3600000000)
            y_cpu.append(m[time]['cpu_usage'])
            y_cpu_percentage.append(m[time]['cpu_usage'] / cpu_available * 100)
        ax.plot(x_time, y_cpu_percentage, label=m_id)

        count += 1
        if count % 4 == 0:
            plt.legend()
    plt.show()

    return


def mem_usage_plt(res_dict, start, end):
    count = 0
    for m_id in res_dict.keys():
        if count % 4 == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.xlabel('Time (h)')
            plt.ylabel('Memory usage (%)')
            plt.xlim((start - start) / 3600000000, (end - start) / 3600000000)

        m = res_dict[m_id]
        mem_available = machine_index[m_id]['mem_available']

        x_time = []
        y_mem = []
        y_mem_percentage = []
        for time in m.keys():
            x_time.append((time - start) / 3600000000)
            y_mem.append(m[time]['mem_usage'])
            y_mem_percentage.append(m[time]['mem_usage'] / mem_available * 100)
        ax.plot(x_time, y_mem_percentage, label=m_id)

        count += 1
        if count % 4 == 0:
            plt.legend()
    plt.show()

    return


# Platform ID usage within days
def analyse_platform_usage(start, end, moments, segment):
    sample_moments_iterator = iter(moments)
    current_sample_moment = next(sample_moments_iterator)
    moment_ind = 0

    res_dict = OrderedDict()

    for p in platform_ids:
        res_dict[p] = OrderedDict([])
        p_dict = res_dict[p]
        for s in moments:
            p_dict[s] = OrderedDict()
            p_dict[current_sample_moment]['cpu_usage'] = 0
            p_dict[current_sample_moment]['total_cpu'] = 0
            p_dict[current_sample_moment]['machines'] = []

    for fn in sorted(listdir('task_usage'))[segment:]:
        print(fn)
        fp = path.join('task_usage', fn)
        trace_df = read_csv(fp, header=None, index_col=False, compression='gzip',
                            names=task_usage_csv_colnames, usecols=task_usage_cols)

        if start <= max(trace_df['starttime']):
            for index, event in trace_df.iterrows():
                if current_sample_moment is None:
                    break

                p_id = machine_index[event['machine_id']]['platform_id']
                p_dict = res_dict[p_id]

                if event['starttime'] <= current_sample_moment <= event['endtime']:
                    p_dict[current_sample_moment]['cpu_usage'] += event['cpu_usage']
                    if event['machine_id'] not in p_dict[current_sample_moment]['machines']:
                        p_dict[current_sample_moment]['machines'].append(event['machine_id'])

                    # Check next sample moments
                    tmp = moment_ind + 1
                    while tmp < len(moments) and event['starttime'] <= moments[tmp] <= event['endtime']:
                        p_dict[moments[tmp]]['cpu_usage'] += event['cpu_usage']
                        if event['machine_id'] not in p_dict[moments[tmp]]['machines']:
                            p_dict[moments[tmp]]['machines'].append(event['machine_id'])
                        tmp += 1

                if event['starttime'] > current_sample_moment:
                    for p_dic in res_dict.values():
                        machine_list = p_dic[current_sample_moment]['machines']
                        for m_id in machine_list:
                            cpu_available = machine_index[m_id]['cpu_available']
                            p_dic[current_sample_moment]['total_cpu'] += cpu_available

                    try:
                        current_sample_moment = next(sample_moments_iterator)
                        moment_ind += 1
                    except StopIteration:
                        current_sample_moment = None
        if max(trace_df['starttime']) > end or current_sample_moment is None:
            break

    return res_dict


def platform_cpu_usage_plt(res_dict, start, end):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for platform in res_dict.keys():
        m = res_dict[platform]
        x_time = []
        y_cpu = []
        for time in m.keys():
            x_time.append((time - start) / 3600000000)
            if m[time]['total_cpu'] == 0:
                res = 0
            else:
                res = m[time]['cpu_usage'] / m[time]['total_cpu'] * 100
            y_cpu.append(res)
        ax.plot(x_time, y_cpu, label=platform)

    plt.xlabel('Time (h)')
    plt.ylabel('CPU usage per machine averaged across platform ID (%)')
    plt.xlim((start - start) / 3600000000, (end - start) / 3600000000)
    plt.legend()
    plt.show()

    return


def machine_cross_correlation(series_dict):
    start = snapshot_start

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('Time shift (min)')
    plt.ylabel('Cross correlation coefficient between two machines time series CPU usage')
    plt.ylim(-1, 1)

    # shift_sample = int(len(sample_moments) / 2)
    shift_sample = 50
    res_acc = [0] * shift_sample

    keys = []
    for k in series_dict.keys():
        keys.append(k)

    for k in range(0, num_sample_machines, 2):
        try:
            m_id = keys[k]
            n_id = keys[k+1]
        except IndexError:
            break

        m = series_dict[m_id]
        n = series_dict[n_id]
        cpu_available_m = machine_index[m_id]['cpu_available']
        cpu_available_n = machine_index[n_id]['cpu_available']

        x_time = []
        y_cpu_percentage_m = []
        y_cpu_percentage_n = []

        for time in m.keys():
            x_time.append((time - start) / 3600000000)
            y_cpu_percentage_m.append(m[time]['cpu_usage'] / cpu_available_m * 100)

        for time in n.keys():
            # x_time_n.append((time - start) / 3600000000)
            y_cpu_percentage_n.append(n[time]['cpu_usage'] / cpu_available_n * 100)

        label = str(m_id) + ',' + str(n_id)
        # Compute correlation between m and n cpu usage percentage (time-shift)
        tau_acc = []
        res = []
        tau_shift = interval / 60000000

        for tau in range(0, shift_sample):
            tau_acc.append(tau * tau_shift)

            a = y_cpu_percentage_m[0:shift_sample]
            a_shift = y_cpu_percentage_n[tau:shift_sample + tau]

            n = len(a)
            a_mean = np.mean(a)
            a_shift_mean = np.mean(a_shift)

            corr_1 = np.sum(np.multiply(a, a_shift)) - n * a_mean * a_shift_mean
            corr_2 = np.sqrt(np.sum(np.square(a)) - n * a_mean ** 2) * np.sqrt(
                np.sum(np.square(a_shift)) - n * a_shift_mean ** 2)
            corr = corr_1 / corr_2
            if np.isnan(corr):
                corr = 0

            res.append(corr)
            res_acc[tau] += corr
        ax.plot(tau_acc, res, label=label)
    plt.legend()
    plt.show()

    r_inc = 0
    for r in res_acc:
        res_acc[r_inc] = r / (num_sample_machines / 2)
        r_inc += 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('Time shift (min)')
    plt.ylabel('Average cross correlation coefficient between two machines time series CPU usage')
    ax.plot(tau_acc, res_acc)
    plt.show()
    return


def machine_auto_correlation(series_dict):
    start = snapshot_start

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('Time shift (min)')
    plt.ylabel('Auto correlation coefficient per machine time series CPU usage')
    plt.ylim(-1, 1)

    # shift_sample = int(len(y_cpu_percentage) / 2)
    shift_sample = 120
    res_acc = [0] * shift_sample

    for m_id in series_dict.keys():
        m = series_dict[m_id]
        cpu_available_m = machine_index[m_id]['cpu_available']

        x_time = []
        y_cpu_percentage = []

        for time in m.keys():
            x_time.append((time - start) / 3600000000)
            y_cpu_percentage.append(m[time]['cpu_usage'] / cpu_available_m * 100)

        # Compute correlation between m and time-shifted m
        tau_acc = []
        res = []
        tau_shift = interval / 60000000

        for tau in range(0, shift_sample):
            tau_acc.append(tau * tau_shift)

            a = y_cpu_percentage[0:shift_sample]
            a_shift = y_cpu_percentage[tau:shift_sample + tau]

            n = len(a)
            a_mean = np.mean(a)
            a_shift_mean = np.mean(a_shift)

            corr_1 = np.sum(np.multiply(a, a_shift)) - n * a_mean * a_shift_mean
            corr_2 = np.sqrt(np.sum(np.square(a)) - n * a_mean ** 2) * np.sqrt(
                np.sum(np.square(a_shift)) - n * a_shift_mean ** 2)

            corr = corr_1 / corr_2

            if np.isnan(corr):
                corr = 0
            res.append(corr)
            res_acc[tau] += corr
        ax.plot(tau_acc, res, label=m_id)
    plt.legend()
    plt.show()

    r_inc = 0
    for r in res_acc:
        res_acc[r_inc] = r / num_sample_machines
        r_inc += 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('Time shift (min)')
    plt.ylabel('Average auto correlation coefficient over ' + str(num_sample_machines) + ' machines')
    ax.plot(tau_acc, res_acc)
    plt.show()
    return


def reduce_task_usage_parse(start):
    segment = 0
    step = 10

    for fn in sorted(listdir('task_usage'))[step:500:step]:
        fp = path.join('task_usage', fn)
        trace_df = read_csv(fp, header=None, index_col=False, compression='gzip',
                            names=task_usage_csv_colnames, usecols=task_usage_cols)
        if start <= max(trace_df['starttime']):
            break
        else:
            segment += step

    return segment


def save_series_dict_csv(series_dict):
    for k in series_dict.keys():
        m = series_dict[k]
        cpu_available = machine_index[k]['cpu_available']
        mem_available = machine_index[k]['mem_available']

        if path.isfile('../compiled_data_ssampling_test/' + str(k) + '_cpu_usage.csv'):
            continue

        with open('../compiled_data_ssampling_test/' + str(k) + '_cpu_usage.csv', "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for time in m.keys():
                cpu = m[time]['cpu_usage']
                percent = (cpu / cpu_available) * 100
                csv_writer.writerow([time, cpu, percent])

        with open('../compiled_data_ssampling_test/' + str(k) + '_mem_usage.csv', "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for time in m.keys():
                mem = m[time]['mem_usage']
                percent = (mem / mem_available) * 100
                csv_writer.writerow([time, mem, percent])

    return


def save_machine_index():
    with open('../machine_index.csv', "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for k in machine_index.keys():
            platform_id = machine_index[k]['platform_id']
            cpu_available = machine_index[k]['cpu_available']
            mem_available = machine_index[k]['mem_available']
            csv_writer.writerow([k, platform_id, cpu_available, mem_available])

    return


chdir('clusterdata-2011-2')

# 06/05/19
# ------- MACHINE SPECIFIC ANALYSIS -------
# Sort by machine ID
# sample a few machine IDs by random

machine_trace_path = 'part-00000-of-00001.csv.gz'
machine_event_time_df = read_csv(path.join('machine_events', machine_trace_path), header=None, index_col=False,
                                 compression='gzip', names=machine_events_csv_colnames)

# sample moments to take from a certain time period (defined to be a week or day or past 5 min)
# 86400000000 # day
# 43200000000 # 12 hours
# 21600000000 # 6 hours
# 604800000000  # weekly
# 3600000000  # hourly
# 300000000 # 5 min
max_time = max(machine_event_time_df['time'])
period = 21600000000
snapshot_start = randint(0, max_time)
snapshot_end = snapshot_start + period

print("Machine analysis period -", str(snapshot_start), ":", str(snapshot_end))

platform_ids = []
for i, e in machine_event_time_df.iterrows():
    if e['platform_id'] not in platform_ids:
        platform_ids.append(e['platform_id'])
print(platform_ids)
num_platforms = len(platform_ids)

machine_ids = []
for i, e in machine_event_time_df.iterrows():
    if e['machine_id'] not in machine_index:
        machine_ids.append(e['machine_id'])

        machine_index[e['machine_id']] = OrderedDict()
        machine_index[e['machine_id']]['platform_id'] = e['platform_id']
        machine_index[e['machine_id']]['cpu_available'] = e['cpu']
        machine_index[e['machine_id']]['mem_available'] = e['memory']

num_machines = len(machine_index)
sample_machine_ids = []
# sample_machine_ids = [348805021, 350588109, 4820285492, 1436333635,
#                       3338000908, 1390835522, 1391018274, 5015788232, 4874102959]
option_check = True
check_id = 0

if len(sample_machine_ids) == 0:
    for x in range(num_sample_machines):
        platform_check = False
        while not platform_check:
            sample_index = randint(0, num_machines - 1)

            if not option_check:
                sample_machine_ids.append(machine_ids[sample_index])
                platform_check = True
            # force check platform ID
            elif option_check and machine_index[machine_ids[sample_index]]['platform_id'] == platform_ids[check_id]:
                sample_machine_ids.append(machine_ids[sample_index])
                platform_check = True
print(sample_machine_ids)

sample_moments = []
interval = 60000000
for t in range(snapshot_start, snapshot_end + interval, interval):
    sample_moments.append(t)

starting_trace = reduce_task_usage_parse(snapshot_start)

# platform_series_data = analyse_platform_usage(snapshot_start, snapshot_end, sample_moments, starting_trace)
machine_series_data = analyse_machine_usage(snapshot_start, snapshot_end,
                                            sample_moments, sample_machine_ids, starting_trace)

# Uncomment functions to run them

# Resource usage plots
# cpu_usage_plt(machine_series_data, start, end)
# mem_usage_plt(machine_series_data, start, end)
# platform_cpu_usage_plt(platform_series_data, start, end)

# Correlation analysis
machine_auto_correlation(machine_series_data)
# machine_cross_correlation(machine_series_data)

# Generate timeseries data
# save_series_dict_csv(machine_series_data)
# save_machine_index()
