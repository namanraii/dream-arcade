from dreamarcade.data.detectors import ball_count, detect_balls, detect_paddle
from dreamarcade.data.env import OriginalBreakout, OriginalPong
from dreamarcade.data.policies import ScriptedPolicy
from dreamarcade.eval.physics import physics_report


def test_up_moves_the_paddle_toward_smaller_y():
    env = OriginalPong(seed=0, max_steps=20)
    env.reset()
    y0 = env.agent_y
    env.step(2)
    assert env.agent_y < y0
    env.step(3)
    env.step(3)
    assert env.agent_y > y0 - 2.4


def test_one_ball_and_a_detectable_paddle():
    env = OriginalPong(seed=1, max_steps=30)
    frame, _ = env.reset()
    assert frame.shape == (64, 64, 3)
    assert ball_count(frame) == 1
    paddle = detect_paddle(frame, side="right")
    assert paddle is not None
    assert abs(paddle - (env.agent_y + env.paddle_h / 2)) < 3


def test_wall_reflection_flips_vertical_velocity():
    env = OriginalPong(seed=0, max_steps=10)
    env.reset()
    env.waiting = False
    env.ball_x = 30
    env.ball_y = 1.2
    env.vx = 0.4
    env.vy = -1.0
    env._move_ball()
    assert env.vy > 0


def test_scripted_policy_serves_and_tracks():
    env = OriginalPong(seed=2, max_steps=40)
    obs, _ = env.reset()
    policy = ScriptedPolicy()
    assert policy.act(obs, env) == 1
    frames = [obs]
    for _ in range(40):
        obs, _reward, done, _info = env.step(policy.act(obs, env))
        frames.append(obs)
        if done:
            break
    report = physics_report(frames)
    assert report["single_ball_fraction"] > 0.9


def test_breakout_action_moves_horizontally():
    env = OriginalBreakout(seed=0, max_steps=10)
    env.reset()
    x0 = env.paddle_x
    env.step(2)
    assert env.paddle_x > x0
    frame = env.render()
    blobs = detect_balls(frame, ignore_columns=(0, 64))
    assert len(blobs) >= 1


def test_detector_ignores_the_paddles_as_balls():
    env = OriginalPong(seed=0)
    frame, _ = env.reset()
    blobs = detect_balls(frame)
    assert len(blobs) == 1
    _cy, cx, _area = blobs[0]
    assert 6 < cx < 58
