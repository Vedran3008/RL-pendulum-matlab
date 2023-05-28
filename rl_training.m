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
observationInfo = rlNumericSpec([4 1], 'LowerLimit', -1000*ones(4,1), 'UpperLimit', 1000*ones(4,1));
observationInfo.Name = 'states';
actionInfo = rlNumericSpec([1 1], 'LowerLimit', -10, 'UpperLimit', 10);
actionInfo.Name = 'power';
blk = [mdl '/RLAgent']; % Replace 'RL Agent' with the name of your RL Agent block
env = rlSimulinkEnv(mdl, blk, observationInfo, actionInfo);
whos('env')

criticLayerSizes = 50;
criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
critic = buildCritic(criticLayerSizes,observationInfo,actionInfo,criticOpts);
actorLayerSizes = 50;
actorOpts = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);
actor = buildActor(actorLayerSizes,observationInfo,actionInfo,actorOpts);
agentOpts = rlDDPGAgentOptions('SampleTime',0.01,'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6,'DiscountFactor',0.99);
agent = rlDDPGAgent(actor,critic,agentOpts);

trainingOpts = rlTrainingOptions('MaxEpisodes',500,'MaxStepsPerEpisode',1000);
trainResults = train(agent, env, trainingOpts);





function actor = buildActor(actorLayerSizes,observationInfo,actionInfo,actorOpts)
    % Define the actor network
    actorNetwork = [
        featureInputLayer(observationInfo.Dimension(1),'Normalization','none','Name','state')
        fullyConnectedLayer(actorLayerSizes,'Name','fc')
        reluLayer('Name','relu')
        fullyConnectedLayer(actionInfo.Dimension(1),'Name','action')];
    actorNetwork = layerGraph(actorNetwork);

    % Create the actor representation
    actor = rlDeterministicActorRepresentation(actorNetwork,observationInfo,actionInfo,'Observation',{'state'},'Action',{'action'},actorOpts);
end

function critic = buildCritic(criticLayerSizes,observationInfo,actionInfo,criticOpts)
    % Define the critic network
    statePath = [
        featureInputLayer(observationInfo.Dimension(1),'Normalization','none','Name','state')
        fullyConnectedLayer(criticLayerSizes,'Name','stateFC')];
    actionPath = [
        featureInputLayer(actionInfo.Dimension(1),'Normalization','none','Name','action')
        fullyConnectedLayer(criticLayerSizes,'Name','actionFC')];
    commonPath = [
        additionLayer(2,'Name','add')
        reluLayer('Name','relu')
        fullyConnectedLayer(1,'Name','Q')];
    
    criticNetwork = layerGraph(statePath);
    criticNetwork = addLayers(criticNetwork, actionPath);
    criticNetwork = addLayers(criticNetwork, commonPath);

    criticNetwork = connectLayers(criticNetwork,'stateFC','add/in1');
    criticNetwork = connectLayers(criticNetwork,'actionFC','add/in2');
    
    % Create the critic representation
    critic = rlQValueRepresentation(criticNetwork,observationInfo,actionInfo,'Observation',{'state'},'Action',{'action'},criticOpts);
end


