function runTest

    % To be run only once from the EE368_proj directory
    
    % Number of Test images ('testXX.jpg' XX = 1-20)
    for i = 1:20
        disp(sprintf('Testing image %d of 20', i))
        imgFolder = ['Testing\test',num2str(i),'\'];
        imgPath = [imgFolder,'test',num2str(i),'.jpg'];
        
        % Test image 10 times
        for j = 1:10
            disp(sprintf('--->Test: %d',j))
            outPath = [imgFolder,'test',num2str(i),'_out_',num2str(j),'.jpg'];
            matchImagesMultiOptimized(imgPath,outPath);
        end
        disp(sprintf('\n'))
    end
end
