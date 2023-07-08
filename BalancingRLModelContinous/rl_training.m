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
    

Tsample = 0.01;
Tsim = 10;

rng(0)

mdl = 'RL_nelinearni_model_njihalaCont'; % Replace with your model's name
load_system(mdl)
isLoaded = bdIsLoaded('RL_nelinearni_model_njihalaCont');
if isLoaded
    disp('Model is loaded');
else
    disp('Model is not loaded');
end

obsInfo = rlNumericSpec([4 1], 'LowerLimit', -1000*ones(4,1), 'UpperLimit', 1000*ones(4,1));
obsInfo.Name = 'states';

actInfo = rlNumericSpec([1 1], 'LowerLimit', -3, 'UpperLimit', 3);
actInfo.Name = 'power';


blk = [mdl '/RLAgent']; % Replace 'RL Agent' with the name of your RL Agent block
env = rlSimulinkEnv(mdl, blk, obsInfo, actInfo);
whos('env')


% Define state path
statePath = [
    featureInputLayer( ...
        obsInfo.Dimension(1), ...
        Name="obsPathInputLayer")
    fullyConnectedLayer(400)
    reluLayer
    fullyConnectedLayer(300)
    reluLayer
    fullyConnectedLayer(300,Name="spOutLayer")
    ];

% Define action path
actionPath = [
    featureInputLayer( ...
        actInfo.Dimension(1), ...
        Name="actPathInputLayer")
    fullyConnectedLayer(300, ...
        Name="actMiddleLayer", ...
        BiasLearnRateFactor=0)
     fullyConnectedLayer(300, ...
        Name="apOutLayer", ...
        BiasLearnRateFactor=0)
    ];

% Define common path
commonPath = [
    additionLayer(2,Name="add")
    reluLayer
    fullyConnectedLayer(1)
    ];

% Create layergraph, add layers and connect them
criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,"spOutLayer","add/in1");
criticNetwork = connectLayers(criticNetwork,"apOutLayer","add/in2");

critic = rlQValueFunction(criticNetwork, ...
    obsInfo,actInfo, ...
    ObservationInputNames="obsPathInputLayer", ...
    ActionInputNames="actPathInputLayer");


%ACTOR
actorNetwork = [
    featureInputLayer(obsInfo.Dimension(1))
    fullyConnectedLayer(400)
    reluLayer
    fullyConnectedLayer(300)
    reluLayer
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(1)
    tanhLayer
    scalingLayer(Scale=max(actInfo.UpperLimit))
    ];
actorNetwork = dlnetwork(actorNetwork);
actor = rlContinuousDeterministicActor(actorNetwork,obsInfo,actInfo);


criticOpts = rlOptimizerOptions(LearnRate=1e-03,GradientThreshold=1);
actorOpts = rlOptimizerOptions(LearnRate=1e-04,GradientThreshold=1);

agentOpts = rlDDPGAgentOptions(...
    SampleTime=Tsample,...
    CriticOptimizerOptions=criticOpts,...
    ActorOptimizerOptions=actorOpts,...
    ExperienceBufferLength=1e6,...
    DiscountFactor=0.99,...
    MiniBatchSize=128);

agentOpts.NoiseOptions.Variance = 0.6;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-6;

agent = rlDDPGAgent(actor,critic,agentOpts);

maxepisodes = 5000;
maxsteps = ceil(Tsim/Tsample);
trainOpts = rlTrainingOptions(...
    MaxEpisodes=maxepisodes,...
    MaxStepsPerEpisode=maxsteps,...
    ScoreAveragingWindowLength=10,...
    Verbose=true,...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=700,...
    SaveAgentCriteria="EpisodeReward",...
    SaveAgentValue=500);


doTraining = true;
if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load the pretrained agent for the example.
    load("SimulinkPendulumDDPG.mat","agent")
end


function resetEnvironment()
    % Reset your variables here
    global myVariable
    myVariable = 0;
    % Continue for all other variables...
end

