# dynamic-follow-tf-v3

Just a testing ground for experimenting with TensorFlow and applying it to a longitudinal control system for openpilot (but v2!)

This version includes predicting the brake position based off a Keras model that was trained on data from the Holden Astra with a hyper-accurate brake sensor data (something my Corolla lacks). Where previously I was just subtracting from the output in openpilot to get fake-braking, now all brake samples are based off `v_ego` and `a_ego`.

This model is named `live_tracks`, as it solely uses up to 16 live radar points, v_ego, steer_rate, steer_angle, left and right blinkers to predict whether to accelerate, coast, or brake. All without relying on openpilot's vision system to pick out a lead. With enough data and training, this should allow the model to behave proactively on the road, not reactively.

Some cases where the model has already demonstrated superior performance in the real world:

- Approaching a tight bend in the road, passing through a narrow railroad crossing, the model slowed me down by 10 mph and took the curve. As soon as the wheel started to straighten out, it reaccelerated.
- I flicked on my turn signal, only for the model to smoothly start braking. I made the 90° turn perfectly, and as I was straightening the wheel out, it reaccelerated.
- Approaching two stopped cars at a light, the model coasted until the light turned green and the cars started to pick back up. This was all while openpilot showed no leads ahead on-screen.

Some cases where the model could demonstrate superior performance in the real world (not enough data yet):

- Since most radars can bounce under the lead vehicle to get the car ahead of the lead (and even more than one object), the model can know when traffic ahead of the lead is starting to slow and immedately start making a larger following distance.
- Say you're on the highway following a car at your desired speed with a safe distance. A car in the adjacent lane (significantly slower than you) starts to change into you. With up to the 16 radar points the model can see at any one time, it always knows the relative speed and relative distance longitudinally (and laterally) of each car ahead of you. It can can then decide to brake, or take no action if it poses no threat (far enough away or a higher relative velocity).

<img src="left_turn.gif?raw=true" width="800">
