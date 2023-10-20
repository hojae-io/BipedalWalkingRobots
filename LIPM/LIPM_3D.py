# -*-coding:UTF-8 -*-
import numpy as np

# Deinition of 3D Linear Inverted Pendulum
class LIPM3D:
    def __init__(self,
                 dt=0.001,
                 T_sup=1.0,
                 support_leg='left_leg'):
        self.dt = dt
        self.t = 0
        self.T_sup = T_sup # support time

        self.p_x = 0  # desired foot location x
        self.p_y = 0  # desired foot location y

        self.p_x_star = 0 # modified foot location x
        self.p_y_star = 0 # modified foot location y

        # Initialize the gait parameters
        self.s_x = 0.0
        self.s_y = 0.0

        # COM initial state
        self.x_0 = 0
        self.vx_0 = 0
        self.y_0 = 0
        self.vy_0 = 0

        # COM real-time state
        self.x_t = 0
        self.vx_t = 0
        self.y_t = 0
        self.vy_t = 0

        # COM desired state
        self.x_d = 0
        self.vx_d = 0
        self.y_d = 0
        self.vy_d = 0

        # final state for one gait unit
        self.bar_x = 0.0
        self.bar_y = 0.0
        self.bar_vx = 0.0
        self.bar_vy = 0.0

        self.support_leg = support_leg
        self.left_foot_pos = [0.0, 0.0, 0.0]
        self.right_foot_pos = [0.0, 0.0, 0.0]
        self.COM_pos = [0.0, 0.0, 0.0]
    
    def initializeModel(self, COM_pos, left_foot_pos, right_foot_pos):
        self.COM_pos = COM_pos
        self.left_foot_pos = left_foot_pos
        self.right_foot_pos = right_foot_pos

        self.zc = self.COM_pos[2]
        self.w_0 = np.sqrt(self.zc / 9.81) # set gravity parameter as 9.81 = w_0
        self.C = np.cosh(self.T_sup/self.w_0)
        self.S = np.sinh(self.T_sup/self.w_0)
    
    def updateParameters(self, T_sup):
        self.T_sup = T_sup
        self.C = np.cosh(self.T_sup/self.w_0)
        self.S = np.sinh(self.T_sup/self.w_0)

    def step(self):
        self.t += self.dt
        t = self.t
        w_0 = self.w_0

        self.x_t = self.x_0*np.cosh(t/w_0) + w_0*self.vx_0*np.sinh(t/w_0)
        self.vx_t = self.x_0/w_0*np.sinh(t/w_0) + self.vx_0*np.cosh(t/w_0)

        self.y_t = self.y_0*np.cosh(t/w_0) + w_0*self.vy_0*np.sinh(t/w_0)
        self.vy_t = self.y_0/w_0*np.sinh(t/w_0) + self.vy_0*np.cosh(t/w_0)

    def calculateXtVt(self, t):
        """ Calculate next step location """
        w_0 = self.w_0

        x_t = self.x_0*np.cosh(t/w_0) + w_0*self.vx_0*np.sinh(t/w_0)
        vx_t = self.x_0/w_0*np.sinh(t/w_0) + self.vx_0*np.cosh(t/w_0)

        y_t = self.y_0*np.cosh(t/w_0) + w_0*self.vy_0*np.sinh(t/w_0)
        vy_t = self.y_0/w_0*np.sinh(t/w_0) + self.vy_0*np.cosh(t/w_0)

        return x_t, vx_t, y_t, vy_t

    def calculateFootLocationForNextStepXcoM(self, s=0.5, w=0.4): # s: desired step length, w: desired step width
        ICP_x = self.x_t + self.vx_t*self.w_0
        ICP_y = self.y_t + self.vy_t*self.w_0
        b_x = s / (np.exp(self.T_sup/self.w_0) - 1)
        b_y = w / (np.exp(self.T_sup/self.w_0) + 1)
        # self.p_x_star = self.p_x + 0.4
        self.p_x_star = ICP_x - b_x
        self.p_y_star = ICP_y + b_y if self.support_leg == "left_leg" else ICP_y - b_y

        return self.p_x_star, self.p_y_star
        # self.p_y_star = -0.3 if self.support_leg == "left_leg" else 0.3

    def switchSupportLeg(self):
        if self.support_leg == 'left_leg':
            print('\n---- switch the support leg to the right leg')
            self.support_leg = 'right_leg'
            COM_pos_x = self.x_t + self.left_foot_pos[0]
            COM_pos_y = self.y_t + self.left_foot_pos[1]
            self.x_0 = COM_pos_x - self.right_foot_pos[0]
            self.y_0 = COM_pos_y - self.right_foot_pos[1]
        elif self.support_leg == 'right_leg':
            print('\n---- switch the support leg to the left leg')
            self.support_leg = 'left_leg'
            COM_pos_x = self.x_t + self.right_foot_pos[0]
            COM_pos_y = self.y_t + self.right_foot_pos[1]
            self.x_0 = COM_pos_x - self.left_foot_pos[0]
            self.y_0 = COM_pos_y - self.left_foot_pos[1]

        self.t = 0
        self.vx_0 = self.vx_t
        self.vy_0 = self.vy_t
