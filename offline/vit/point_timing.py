def resume(saved,now):
    if saved is None:return dict(active_seconds=0.,first_started=now,attempt_started=now,pause_wait_seconds=0.,phase_seconds={})
    return dict(active_seconds=saved['active_seconds'],first_started=saved['first_started'],attempt_started=now,
                pause_wait_seconds=saved['pause_wait_seconds']+max(0.,now-saved['paused_at']),phase_seconds=dict(saved['phase_seconds']))

def checkpoint(state,phases,now):
    return dict(active_seconds=state['active_seconds']+now-state['attempt_started'],first_started=state['first_started'],paused_at=now,
                pause_wait_seconds=state['pause_wait_seconds'],phase_seconds=dict(phases))

def finish(state,now):
    return dict(seconds=state['active_seconds']+now-state['attempt_started'],pause_wait_seconds=state['pause_wait_seconds'],wall_span_seconds=now-state['first_started'])
