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

statePath = [
    featureInputLayer(numObservations,'Normalization','none','Name','observation')
    fullyConnectedLayer(24, 'Name', 'CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(24, 'Name', 'CriticStateFC2')];

actionPath = [
    featureInputLayer(numActions,'Normalization','none','Name','action')
    fullyConnectedLayer(24, 'Name', 'CriticActionFC1')];

commonPath = [
    additionLayer(2,'Name', 'add')
    reluLayer('Name', 'CriticCommonRelu')
    fullyConnectedLayer(1, 'Name', 'output')];

% Create separate layer graphs for state and action paths
statePathGraph = layerGraph(statePath);
actionPathGraph = layerGraph(actionPath);

% Combine state and action paths
lgraph = addLayers(statePathGraph, actionPathGraph.Layers);
lgraph = addLayers(lgraph, commonPath);
lgraph = connectLayers(lgraph, 'CriticStateFC2', 'add/in1');
lgraph = connectLayers(lgraph, 'CriticActionFC1', 'add/in2');

criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
critic = rlQValueRepresentation(lgraph, obsInfo, actInfo, 'Observation',{'observation'}, 'Action',{'action'}, criticOpts);




actorNet = [
    featureInputLayer(prod(obsInfo.Dimension),'Name','observations')
    fullyConnectedLayer(50,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(50,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(50,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(50,'Name','fc4')
    reluLayer('Name','relu4')
    fullyConnectedLayer(50,'Name','fc5')
    reluLayer('Name','relu5')
    fullyConnectedLayer(50,'Name','fc6')
    reluLayer('Name','relu6')
    fullyConnectedLayer(1,'Name','fc7')
    tanhLayer('Name','voltage')
];


actorNet = dlnetwork(actorNet);

actor = rlDeterministicActorRepresentation(actorNet, obsInfo, actInfo, 'Observation', {'observations'}, 'Action', {'voltage'}, scalingLayer);

summary(actorNet)



%actor = rlDeterministicActorRepresentation(actorNet,obsInfo,actInfo);

% Create the agent
agentOpts = rlDDPGAgentOptions(...
    'SampleTime', Ts, ...
    'TargetSmoothFactor', 1e-3, ...
    'DiscountFactor', 0.99, ...
    'MiniBatchSize', 64, ...
    'ExperienceBufferLength', 1e6, ...  % Maximum size of experience buffer
    'SaveExperienceBufferWithAgent', false, ...  % Save experience buffer with agent
    'ResetExperienceBufferBeforeTraining', false, ...  % Do not clear the buffer during training
    'NumStepsToLookAhead', 1, ...
    'ResetExperienceBufferBeforeTraining',true);
agent = rlDDPGAgent(actor,critic,agentOpts);


trainingOpts = rlTrainingOptions('MaxEpisodes',5000,'MaxStepsPerEpisode',10000, ...
    'Plots','training-progress', ...
    'Verbose',false);
trainResults = train(agent, env, trainingOpts);


simOptions = rlSimulationOptions('MaxSteps', 500);
experience = sim(env, agent, simOptions);
