classdef CustomPendulumEnv < rl.env.MATLABEnvironment
    properties
        % The name of your Simulink model
        ModelName
        % Simulation parameters
        Ts = 0.001;  % Sample time

        % parametri za simulaciju nelinearnog modela njihala
        J0_kapa = 5.5351*10^(-4);
        J1_kapa = 3.4356*10^(-4);
        J2_kapa = 3.8533*10^(-4);
        
        b1 = 8.3336*10^(-5);
        b2 = 2.5*10^(-4);
        g = 9.81;
        
        m2 = 18.1*10^(-3);
        L1 = 10.85*10^(-2);
        L2 = 18.2*10^(-2);
        l2 = 13.8*10^(-2);
        
        Ra = 2.19;
        La = 278*10^(-6);
        ce = 0.02559;
        cm = 0.02559;
        n = 3.9;

    end
    
    methods
        function this = CustomPendulumEnv(modelName)
            % Define observation info and action info
            % Observation space: [theta, dtheta, alpha, dalpha ]
            observationInfo = rlNumericSpec([4 1], 'LowerLimit', [-inf; -inf; -inf; -inf], 'UpperLimit', [inf; inf; inf; inf]);
            observationInfo.Name = 'observations';
            % Action space: voltage
            %actionInfo = rlNumericSpec([1 1], 'LowerLimit', -10, 'UpperLimit', 10);
            actionInfo = rlFiniteSetSpec([-10 10]);
            actionInfo.Name = 'voltage';

            % Call superclass constructor to construct the environment
            this = this@rl.env.MATLABEnvironment(observationInfo, actionInfo);

            this.ModelName = modelName;
        end

        function [nextState, reward, isDone, loggedSignals] = step(this, action)
            % Set the action as a parameter in the Simulink model
            set_param([this.ModelName '/Voltage'], 'Value', num2str(action));

            % Simulate the model for one step
            simOut = sim(this.ModelName, 'StopTime', num2str(this.Ts));

            % Get the new state from the output of the model
            data = simOut.simout(1).Data(2, 1:4);  % Get the data
            theta = data(1);
            dtheta = data(2);
            alpha = data(3);
            dalpha = data(4);
            nextState = [theta, dtheta, alpha, dalpha];
            % Compute reward and isDone
            
            % Set to false for now
            isDone = false;
            
            % Logged signals
            loggedSignals = [];
            
            % Compute reward here or in a separate function
            reward = this.getReward(theta, dtheta, alpha, dalpha, action);
        end

        function initialObs = reset(this)
            % Reset the environment to its initial state
            % ...
            
            % For now we will just return zeros
            initialObs = zeros(4, 1);
            
            % You can set the initial state of the Simulink model using set_param function as we did in step function
        end

        function reward = getReward(this, theta, dtheta, alpha, dalpha, ~)
            % Define your reward function here
            alpha_threshold = pi/3; % Adjust this threshold value based on your requirements
            alpha_distance = abs(alpha); % Calculate the absolute distance from zero
            
            if alpha_distance <= alpha_threshold % Check if alpha is within the threshold range
                reward =  pi- abs(alpha); 
            else
                reward = 0; % No reward if alpha is outside the threshold range
            end

        end
    end
end
