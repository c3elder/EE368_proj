function results = vl_test_pegasos(varargin)
% VL_TEST_KDTREE
vl_test_init ;

function s = setup()
randn('state',0) ;

s.biasMultiplier = 10 ;
s.lambda = 0.01 ;

Np = 10 ;
Nn = 10 ;
Xp = diag([1 3])*randn(2, Np) ;
Xn = diag([1 3])*randn(2, Nn) ;
Xp(1,:) = Xp(1,:) + 2 + 1 ;
Xn(1,:) = Xn(1,:) - 2 + 1 ;

s.X = [Xp Xn] ;
s.y = [ones(1,Np) -ones(1,Nn)] ;
%s.w = exact_solver(s.X, s.y, s.lambda, s.biasMultiplier)
s.w = [1.181106685845652 ;
       0.098478251033487 ;
       -0.154057992404545 ] ;

function test_problem_1(s)
for conv = {@single,@double}
  vl_twister('state',0) ;
  conv = conv{1} ;
  [w b info] = vl_pegasos(conv(s.X), int8(s.y), s.lambda, ...
                 'MaxIterations', 100000, ...
                 'BiasMultiplier', s.biasMultiplier, ...
                 'BiasLearningRate', .1) ;
  
  % test input
  vl_assert_equal(info.biasMultiplier,s.biasMultiplier); 
  vl_assert_almost_equal(info.biasLearningRate,.1,1e-3); 
  vl_assert_almost_equal(conv([w; b]), conv(s.w), 0.1) ;
end

function test_continue_training(s)
for conv = {@single,@double}
  conv = conv{1} ;

  vl_twister('state',0) ;
  [w b] = vl_pegasos(conv(s.X), int8(s.y), s.lambda, ...
                 'MaxIterations', 3000, ...
                 'BiasMultiplier', s.biasMultiplier) ;

  vl_twister('state',0) ;
  [w1 b1] = vl_pegasos(conv(s.X), int8(s.y), s.lambda, ...
                 'StartingIteration', 1, ...
                 'MaxIterations', 1500, ...
                  'BiasMultiplier', s.biasMultiplier) ;
  [w2 b2] = vl_pegasos(conv(s.X), int8(s.y), s.lambda, ...
                  'StartingIteration', 1501, ...
                  'StartingModel', w1, ...
                  'StartingBias', b1, ...
                  'MaxIterations', 3000, ...
                  'BiasMultiplier', s.biasMultiplier) ;
  vl_assert_almost_equal([w; b],[w2; b2],1e-7) ;
end

function test_continue_training_with_perm(s)
perm = uint32(randperm(size(s.X,2))) ;
for conv = {@single,@double}
  conv = conv{1} ;

  vl_twister('state',0) ;
  [w b] = vl_pegasos(conv(s.X), int8(s.y), s.lambda, ...
                 'MaxIterations', 3000, ...
                 'BiasMultiplier', s.biasMultiplier, ...
                 'Permutation', perm) ;

  vl_twister('state',0) ;
  [w1 b1] = vl_pegasos(conv(s.X), int8(s.y), s.lambda, ...
                 'StartingIteration', 1, ...
                 'MaxIterations', 1500, ...
                 'BiasMultiplier', s.biasMultiplier, ...
                  'Permutation', perm) ;
  [w2 b2] = vl_pegasos(conv(s.X), int8(s.y), s.lambda, ...
                  'StartingIteration', 1501, ...
                  'StartingModel', w1, ...
                  'StartingBias', b1, ...
                  'MaxIterations', 3000, ...
                  'BiasMultiplier', s.biasMultiplier, ...
                  'Permutation', perm) ;

  vl_assert_almost_equal([w; b],[w2; b2],1e-7) ;
end


function test_homkermap(s)
for conv = {@single,@double}
  vl_twister('state',0) ;
  conv = conv{1} ;
  sxe = vl_homkermap(conv(s.X), 1, 'kchi2', 'gamma', .5) ;
  [we be] = vl_pegasos(sxe, int8(s.y), s.lambda, ...
                 'MaxIterations', 100000, ...
                 'BiasMultiplier', s.biasMultiplier, ...
                 'BiasLearningRate', .1) ;
  vl_twister('state',0) ;
  [w b] = vl_pegasos(s.X, int8(s.y), s.lambda, ...
                 'MaxIterations', 100000, ...
                 'BiasMultiplier', s.biasMultiplier, ...
                 'BiasLearningRate', .1,...
                 'homkermap',1,...
                 'gamma',.5,...
                 'kchi2') ;

  vl_assert_almost_equal([w; b],[we; be], 1e-7) ;
end

function test_diagnostic(s)
for conv = {@single,@double}
  vl_twister('state',0) ;
  conv = conv{1} ;

  x = 0;
  dhandle = @(x,stat) (assert(stat.elapsedTime == 0 || stat.elapsedTime ~= 0)) ;

  [w b] = vl_pegasos(s.X, int8(s.y), s.lambda, ...
                        'MaxIterations', 100000, ...
                        'BiasMultiplier', s.biasMultiplier, ...
                        'BiasLearningRate', .1) ;
  vl_twister('state',0) ;
  [wd bd] = vl_pegasos(s.X, int8(s.y), s.lambda, ...
                          'MaxIterations', 100000, ...
                          'BiasMultiplier', s.biasMultiplier, ...
                          'BiasLearningRate', .1,...
                          'DiagnosticFunction',dhandle,...
                          'DiagnosticCallRef',x) ;

  vl_assert_almost_equal([w; b], [wd; bd], 1e-7) ;
end

function test_epsilon(s)
for conv = {@single,@double}
  vl_twister('state',0) ;
  conv = conv{1} ;

  [w b info] = vl_pegasos(s.X, int8(s.y), s.lambda, ...
                        'MaxIterations', 1000000, ...
                        'BiasMultiplier', s.biasMultiplier, ...
                        'BiasLearningRate', .1) ;
  vl_twister('state',0) ;
  [we be infoe] = vl_pegasos(s.X, int8(s.y), s.lambda, ...
                          'MaxIterations', 1000000, ...
                          'Epsilon',1e-7,...
                          'BiasMultiplier', s.biasMultiplier, ...
                          'BiasLearningRate', .1) ;

  vl_assert_almost_equal([w; b], [we; be], 1e-2) ;
  assert(info.iterations > infoe.iterations);
end

function w = exact_solver(X, y, lambda, biasMultiplier)
N = size(X,2) ;
model = svmtrain(y', [(1:N)' X'*X], sprintf(' -c %f -t 4 ', 1/(lambda*N))) ;
w = X(:,model.SVs) * model.sv_coef ;
w(3) = - model.rho / biasMultiplier ;
format long ;
disp('model w:')
disp(w)
