
 %% ************************************************************************
        % readHTK - just incase you ever want to go backwards
        %**************************************************************************
        function [htkdata,nframes,sampPeriod,sampSize,paramKind] = readHTK_new(filename,byte_order)
            
            if nargin<2
                byte_order = 'be';
            end
            
            fid = fopen(filename,'r',sprintf('ieee-%s',byte_order));
            
            nframes = fread(fid,1,'int32');
            sampPeriod = fread(fid,1,'int32');
            sampSize = fread(fid,1,'int16');
            paramKind = fread(fid,1,'int16');
            
            % read the data
            
            htkdata = fread(fid,nframes*(sampSize/4),'float32');
            htkdata = reshape(htkdata,sampSize/4,nframes);
            fclose(fid);
        end % ------ OF READHTK
