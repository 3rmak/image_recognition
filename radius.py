class Radius:
    def __init__(self, estimate, radius):
        self.estimate = estimate
        self.radius = radius

    def set_radius(self, new_rad):
        try:
            if isinstance(new_rad, Radius):
                if self.estimate < new_rad.estimate:
                    self.estimate = new_rad.estimate
                    self.radius = new_rad.radius
        except TypeError:
            print('radius is not instance of RadiusClass')
