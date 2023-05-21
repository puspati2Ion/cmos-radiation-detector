
# Python program to illustrate 
# saving an operated video
  
# organize imports
import time
# importing the module 
import cv2 
import numpy as np

class cmosdet:

    def __init__(self):

        print ('CMOS radiation detector\n______________________\nSTART\n______________________\n')

    def record(self, no=0, no_of_frame=1000, filename='gray.avi', filelog='log.txt'):
                
        self.filename = filename
        
        print ('----------------------------------------\n')
        print ('*RECORDING ionizing events in movie file format*\n')
        print ('----------------------------------------\n')
        
        start = time.time()
        # reading the vedio 
        source = cv2.VideoCapture(no) 

        # We need to set resolutions. 
        # so, convert them from float to integer. 
        frame_width = int(source.get(3)) 
        frame_height = int(source.get(4))
        size = (frame_width, frame_height)
        #print (size)

        self.gray3D = np.zeros(frame_width*frame_height).reshape(frame_width, frame_height)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        result = cv2.VideoWriter(filename,  
                    fourcc, 
                    30, size, 0) 
          
        # running the loop 
        print ('START :', filename)
        iterNo = 1
        while True: 
            
            #print (iterNo)
            # extracting the frames

            ret, img = source.read() 
              
            # converting to gray-scale
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

            # write to gray-scale 
            result.write(gray)

            # displaying the video 
            cv2.imshow("Live measurement", gray) 
            
            # exiting the loop 
            key = cv2.waitKey(1) 
            #if key == ord("q"):
            if iterNo==no_of_frame:
                end = time.time()
                break
            
            iterNo += 1    
            
        # closing the window 
        cv2.destroyAllWindows() 
        source.release()

        print ('\nno of iteration : ',iterNo)
        print ('time taken      : ',end-start)
        print ('END')
        fw = open(filelog, 'w')

        fw.write('no of iteration : '+str(iterNo))
        fw.write('\ntime taken  : '+str(end-start))

        fw.close()
        
    def display(self, filename='gray.avi', filelog='processlog.txt', frame_width=480, frame_height=640):
        
        self.filename = filename
        
        print ('----------------------------------------\n')
        print ('*DISPLAYING ionizing events from movie file format*\n')
        print ('----------------------------------------\n')

        self.gray3D = np.zeros(frame_width*frame_height).reshape(1,frame_width, frame_height)

        # reading the vedio 
        source = cv2.VideoCapture(filename) 

        # running the loop 
        iterNo = 1

        while True: 
          
            # extracting the frames
            try: 
                ret, img = source.read()
                # converting to gray-scale 
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.gray3D = np.concatenate((self.gray3D, [gray]), axis=0)
            except:
                
                print (self.gray3D.shape)
                print (gray)
                print (self.gray3D)
                print ('END') 
                break
              
             

            # write to gray-scale 
            #result.write(gray)

            # displaying the video 
            cv2.imshow("Playing "+self.filename, gray) 
          
            # exiting the loop
            iterNo += 1
            print (iterNo)
            key = cv2.waitKey(1) 
            #if key == ord("q"):
            '''if iterNo == 1000:
                print (gray)
                break'''

        # closing the window 
        cv2.destroyAllWindows() 
        source.release()

        return (self.gray3D)

    def process(self, filename='gray.avi', filelog='processlog.txt', frame_width=480, frame_height=640):
        
        self.filename = filename
        start = time.time()
               
        print ('----------------------------------------\n')
        print ('*CALCULATING mean, min, max, median for each frame from a measurement movie file*\n')
        print ('*as well as contructing 3D array of the movie*\n')
        print ('----------------------------------------\n')   
                
        print ('processing file: ',filename)
        
        # initialization for extracting parameters: mean, min, max, median
        self.mean = np.array([])   #1
        self.min = np.array([])    #2
        self.max = np.array([])    #3
        self.median = np.array([]) #4

        self.gray3D = np.zeros(frame_width*frame_height).reshape(1,frame_width, frame_height)

        # reading the vedio 
        source = cv2.VideoCapture(filename) 

        # running the loop 
        iterNo = 1
        
        while True: 
            
            
            # extracting the frames
            try:
                #print (iterNo)
                
                ret, img = source.read()
                # converting to gray-scale 
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                self.gray3D = np.concatenate((self.gray3D, [gray]), axis=0)
                #print (self.gray3D.shape)
                
                self.mean = np.append(self.mean,np.mean(gray))
                self.min = np.append(self.min,gray.min())
                self.max = np.append(self.max,gray.max())
                self.median = np.append(self.median,np.median(gray))
                
                if iterNo ==300: # original 300
                    break
               
            except:

                
                #print (self.gray3D[1,:,:].shape) # minus the initial template
                #print (gray)
                #print (self.gray3D)
                print ('END') 
                break
            

            # write to gray-scale 
            #result.write(gray)

            # displaying the video 
            #cv2.imshow("Live", gray) 
          
            # exiting the loop
            
            iterNo += 1
            
            key = cv2.waitKey(1) 

            # closing the window 
        cv2.destroyAllWindows() 
        source.release()
        
        end = time.time()
        '''print ('mean :', self.mean)
        print ('min :', self.min)
        print ('max :', self.max)
        print ('median :', self.median)'''
        
        print ('time :', end-start)
        
        print ("*RETURN {'3D array', 'mean', 'min', 'max' and 'median'}*\n")
        print ('****************************************************************************')
        

        #return (self.gray3D[1:],self.mean,self.min,self.max,self.median) # minus the first template
        # minus the first template for self.gray3D
        return {'3D array':self.gray3D[1:], 'mean':self.mean,\
                'min':self.min, 'max':self.max, 'median':self.median}
        
    
    def processBlank(self, filename='gray.avi', filelog='processlog.txt', frame_width=480, frame_height=640):
        
        start = time.time()
        
        print ('processing file: ',filename)
        
        # initialization for extracting parameters: mean, min, max, median
        self.mean = np.array([])   #1
        self.min = np.array([])    #2
        self.max = np.array([])    #3
        self.median = np.array([]) #4

        self.gray3D = np.zeros(frame_width*frame_height).reshape(1,frame_width, frame_height)
        self.pedestal = np.zeros(frame_width*frame_height).reshape(1,frame_width, frame_height)
        self.rms = np.zeros(frame_width*frame_height).reshape(1,frame_width, frame_height)

        # reading the vedio 
        source = cv2.VideoCapture(filename) 

        # running the loop 
        iterNo = 1
        
        while True: 
            
            
            # extracting the frames
            try:
                #print (iterNo)
                
                ret, img = source.read()
                # converting to gray-scale 
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rms = gray**2
                
                self.gray3D = np.concatenate((self.gray3D, [gray]), axis=0)
                self.pedestal = self.pedestal + gray
                self.rms = self.rms + rms
                #print (self.gray3D.shape)
                
                self.mean = np.append(self.mean,np.mean(gray))
                self.min = np.append(self.min,gray.min())
                self.max = np.append(self.max,gray.max())
                self.median = np.append(self.median,np.median(gray))
                
                if iterNo ==300: # original 300
                    break
               
            except:

                
                print (self.gray3D[1,:,:].shape) # minus the initial template
                #print (gray)
                #print (self.gray3D)
                print ('END') 
                break
            

            # write to gray-scale 
            #result.write(gray)

            # displaying the video 
            #cv2.imshow("Live", gray) 
          
            # exiting the loop
            
            iterNo += 1
            
            key = cv2.waitKey(1) 

            # closing the window 
        cv2.destroyAllWindows() 
        source.release()
        
        self.pedestalAvg = self.pedestal/(iterNo-1) # because iterNo starts with one
        self.rmsAvg = self.rms/(iterNo-1)
        
        end = time.time()
        '''print ('mean :', self.mean)
        print ('min :', self.min)
        print ('max :', self.max)
        print ('median :', self.median)'''
        
        print ('time :', end-start)
        print ("*RETURN {'3D array', 'mean', 'min', 'max', 'median', 'pedestalAvg' and 'rmsAvg'}*\n")
        print ('----------------------------------------\n')
        
        return {'3D array':self.gray3D[1:], 'mean':self.mean,\
                'min':self.min, 'max':self.max, 'median':self.median,\
                'pedestalAvg':self.pedestalAvg, 'rmsAvg':self.rmsAvg}
        
        print ('****************************************************************************')

        return (self.gray3D[1:],self.mean,self.min,self.max,self.median,self.pedestalAvg,self.rmsAvg) # minus the first template

    def stats(self):
        
        print ('gray stats for file :', self.filename)
        print ('----------------------------------------\n')
        print ('shape : ', self.gray3D.shape)
        print ('mean  : ', np.mean(self.gray3D))
        print ('min   : ', self.gray3D.min())
        print ('max   : ', self.gray3D.max())
        print ('median: ', np.median(self.gray3D))  

