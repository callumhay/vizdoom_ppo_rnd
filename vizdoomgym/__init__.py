from vizdoomgym.vizdoomenv import VizdoomEnv

class VizdoomDoomGame(VizdoomEnv):
  def __init__(self, **kwargs):
    super(VizdoomDoomGame, self).__init__(-1, **kwargs)
