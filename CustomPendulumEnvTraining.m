% Simulation parameters
%this needs to match in the custom pendulum
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
        
        
env = CustomPendulumEnv('matematicki_model_njihala'); % Replace 'MySimulinkModel' with your actual Simulink model name


obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);


numObservations = prod(obsInfo.Dimension);
numActions = prod(actInfo.Dimension);

rng(0)

net = [
    featureInputLayer(obsInfo.Dimension(1))
    fullyConnectedLayer(20)
    reluLayer
    fullyConnectedLayer(2)
    ];

net = dlnetwork(net);
summary(net)


critic = rlVectorQValueFunction(net,obsInfo,actInfo);

getValue(critic,{rand(obsInfo.Dimension)})

agent = rlDQNAgent(critic);

getAction(agent,{rand(obsInfo.Dimension)})

agent.AgentOptions.UseDoubleDQN = false;
agent.AgentOptions.TargetSmoothFactor = 1;
agent.AgentOptions.TargetUpdateFrequency = 4;
agent.AgentOptions.ExperienceBufferLength = 1e5;
agent.AgentOptions.MiniBatchSize = 256;
agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-3;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;

trainOpts = rlTrainingOptions(...
    MaxEpisodes=1000, ...
    MaxStepsPerEpisode=500, ...
    Verbose=false, ...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=480); 

doTraining = true;
if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load the pretrained agent for the example.
    load("MATLABCartpoleDQNMulti.mat","agent")
end
