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
    

initAlphaList = [pi, pi/12, -pi/12];


mdl = 'RL_nelinearni_model_njihala_PPO'; % Replace with your model's name
blk = [mdl '/RLAgent']; % Replace 'RL Agent' with the name of your RL Agent block
load_system(mdl)
isLoaded = bdIsLoaded('RL_nelinearni_model_njihala_PPO');
if isLoaded
    disp('Model is loaded');
else
    disp('Model is not loaded');
end

% Define the observation and action spaces
obsInfo = rlNumericSpec([4 1], 'LowerLimit', [-Inf; -Inf; -Inf; -Inf], 'UpperLimit', [Inf; Inf; Inf; Inf]);
obsInfo.Name = 'observations';
actInfo = rlFiniteSetSpec(linspace(-10, 10, 21));
actInfo.Name = 'voltage';

% Define the environment
env = rlSimulinkEnv(mdl, blk, obsInfo, actInfo);

rng(0)


% Define the network options
%numObs = obsInfo.Dimension(1);
%numAct = actInfo.Dimension(1);
%hiddenLayerSize = 128;

criticNet = [
    featureInputLayer(prod(obsInfo.Dimension))
    fullyConnectedLayer(25)
    reluLayer
    fullyConnectedLayer(25)
    reluLayer
    fullyConnectedLayer(25)
    reluLayer
    fullyConnectedLayer(25)
    reluLayer
    fullyConnectedLayer(25)
    reluLayer
    fullyConnectedLayer(25)
    reluLayer
    fullyConnectedLayer(1)
    ];

criticNet = dlnetwork(criticNet);
summary(criticNet)

critic = rlValueFunction(criticNet,obsInfo);

actorNet = [
    featureInputLayer(prod(obsInfo.Dimension))
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(numel(actInfo.Elements))
    ];

actorNet = dlnetwork(actorNet);
summary(actorNet)

actor = rlDiscreteCategoricalActor(actorNet,obsInfo,actInfo);

% Create the agent
agent = rlPPOAgent(actor,critic);
agent.AgentOptions.ExperienceHorizon = 1024;
agent.AgentOptions.DiscountFactor = 0.99;

agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-4;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;
agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-4;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1;

trainingOpts = rlTrainingOptions('MaxEpisodes',5000,'MaxStepsPerEpisode',10000, ...
    'Plots','training-progress', ...
    'Verbose',false);
trainResults = train(agent, env, trainingOpts);

