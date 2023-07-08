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
    



mdl = 'RL_nelinearni_model_njihala'; % Replace with your model's name
load_system(mdl)
isLoaded = bdIsLoaded('RL_nelinearni_model_njihala');
if isLoaded
    disp('Model is loaded');
else
    disp('Model is not loaded');
end
% popravi limite
obsInfo = rlNumericSpec([4 1], 'LowerLimit', -1000*ones(4,1), 'UpperLimit', 1000*ones(4,1));
obsInfo.Name = 'states';
actionSet = [-3,0,3];  % You may adjust the step size to suit your needs
actInfo = rlFiniteSetSpec(actionSet);
actInfo.Name = 'power';

blk = [mdl '/RLAgent']; % Replace 'RL Agent' with the name of your RL Agent block
env = rlSimulinkEnv(mdl, blk, obsInfo, actInfo);
whos('env')

criticNet = [
    featureInputLayer(obsInfo.Dimension(1))
    fullyConnectedLayer(400)
    reluLayer
    fullyConnectedLayer(300)
    reluLayer
    fullyConnectedLayer(1)];
criticNet = dlnetwork(criticNet);
critic = rlValueFunction(criticNet,obsInfo);

actorNet = [
    featureInputLayer(obsInfo.Dimension(1))
    fullyConnectedLayer(400)
    reluLayer
    fullyConnectedLayer(300)
    reluLayer
    fullyConnectedLayer(numel(actInfo.Elements))
    softmaxLayer];
actor = rlDiscreteCategoricalActor(actorNet,obsInfo,actInfo);

agent = rlACAgent(actor,critic);

agent.AgentOptions.EntropyLossWeight = 0.01;
agent.AgentOptions.SampleTime = 0.01;

agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-4;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1;
agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-3;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;


plot(criticNet)
plot(actorNet)


trainOpts = rlTrainingOptions(...
    MaxEpisodes=5000,...
    MaxStepsPerEpisode=10000,...
    Verbose=true,...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=980,...
    SaveAgentCriteria="EpisodeReward",...
    SaveAgentValue=980,...
    ScoreAveragingWindowLength=10);



% Specify the save path and file name
savePath = 'D:/Desktop/FER/DiplomskiRad/njihalo/RL-pendulum-matlab/BalancingRLModelDiscreteWORKS';
agentFileName = 'best_agent';


trainResults = train(agent, env, trainOpts);



