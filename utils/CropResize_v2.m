% Crop and resize images under new file names in a different directory 
% Cosmas 2/10/2020
% Bomera 2/29/2020
% Bomera 3/18/2020  

clear variables;
clc;

sourceFolder = '../sourcefolder';                  % the folder with the images
storeFolder='../storefolder';                      % folder to store the images

% Check to make sure that folder actually exists.  Warn user if it doesn't.
% if ~isfolder(sourceFolder)
%   errorMessage = sprintf('Error: The following folder does not exist:\n%s', sourceFolder);
%   uiwait(warndlg(errorMessage));
%   return;
% end

cd(sourceFolder);
%addpath(storefolder);

imageSizes = [512 400 256];
folderPaths = cell(1,3);                                                   % Pre-allocate memory

i = 0;
for size = imageSizes
    i = i + 1;
    folderName  = sprintf('Cropped_%dx%d', size, size);   
    folderPath = strcat(storeFolder, folderName);                          % Full path name
    mkdir(folderPath);
    % mkdir ../storefolder folderName                                      % Create each size folder.
    addpath(folderPath);
    folderPaths{i} = folderPath;                                           % Store all created folder paths
end

% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(sourceFolder, '*.jpg');                             % Change to whatever pattern you need.
theFiles = dir(filePattern);

% Use any of the three folders to check for already cropped files.
croppedFiles = dir(char(fullfile(folderPaths(1), '*.jpg')));
croppedFilesList = cell(1, 600);    % Preallocate memory

% Probably need unpacking
for k = 1 : length(croppedFiles)
    croppedFilesList{k} = croppedFiles(k).name;
end


for k = 1: length(theFiles)
   baseFileName = theFiles(k).name;
   if ~(sum(strcmp(croppedFilesList, baseFileName)) == 1)
        image = imread(baseFileName); 
        croppedImage = imcrop(image);                         	                  % Use cross hairs to select region to crop, then double click
        for i = 1: length(imageSizes)
            resizedImage = imresize(croppedImage, [imageSizes(i) imageSizes(i)]); % Cropped region is resized to this
            fullFileName = fullfile(folderPaths(i), baseFileName); 		          % Keep original file name
            imwrite(resizedImage, char(fullFileName));
        end
   end
end 
