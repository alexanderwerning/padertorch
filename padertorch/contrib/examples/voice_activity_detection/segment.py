import torch
from paderbox.array import segment_axis
from paderbox.array.padding import pad_axis

# adapted from padercontrib.database.chime5.database
def activity_frequency_to_time(
        frequency_activity,
        stft_window_length,
        stft_shift,
        time_length=None,
):
    frequency_activity = frequency_activity[..., None]
    frequency_activity.expand(*(*frequency_activity.size()[:-1], stft_window_length))

    time_activity = torch.zeros(
        (*frequency_activity.size()[:-2],
         frequency_activity.size()[-2] * stft_shift + stft_window_length - stft_shift)
    )

    # Get the correct view to time_signal
    time_signal_seg = segment_axis(
        time_activity, stft_window_length, stft_shift, end=None
    )

    time_signal_seg[:] = frequency_activity

    if time_length is not None:
        if time_length == time_activity.size()[-1]:
            pass
        elif time_length < time_activity.size()[-1]:
            delta = time_activity.size()[-1] - time_length
            assert delta < stft_window_length - stft_shift, (delta, stft_window_length, stft_shift)
            time_activity = time_activity[..., :time_length]

        elif time_length > time_activity.size()[-1]:
            delta = time_length - time_activity.size()[-1]
            assert delta < stft_window_length - stft_shift, (delta, stft_window_length, stft_shift)

            time_activity = pad_axis(
                time_activity,
                pad_width=(0, delta),
                axis=-1,
            )
        else:
            raise Exception('Can not happen')
        assert time_length == time_activity.size()[-1], (time_length, time_activity.size())

    return time_activity
