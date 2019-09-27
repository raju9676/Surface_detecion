from skimage.feature import greycomatrix,greycoprops
import pandas as pd


proList = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy']
featlist= ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy','Label']
properties =np.zeros(5)
glcmMatrix = []
final=[]


for i in range(len(input_X)):
    img = input_X[i]

    # pyplot.imshow((images[k,:,:]),cmap='gray')
    # pyplot.show()
    #  get properties
    glcmMatrix=(greycomatrix(img, [1], [0], levels=256))

   # print(len(glcmMatrix))
    # get properties
    for j in range(0, len(proList)):
        properties[j]=(greycoprops(glcmMatrix, prop=proList[j]))

    features = np.array([properties[0],properties[1],properties[2],properties[3],properties[4],output_Y[i]])
    #print(features)
    final.append(features)

df = pd.DataFrame(final,columns=featlist)
df.to_excel('/content/drive/My Drive/features.xlsx')
from skimage.feature import greycomatrix,greycoprops
import pandas as pd


proList = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy']
featlist= ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy','Label']
properties =np.zeros(5)
glcmMatrix = []
final=[]


for i in range(len(input_X)):
    img = input_X[i]

    # pyplot.imshow((images[k,:,:]),cmap='gray')
    # pyplot.show()
    #  get properties
    glcmMatrix=(greycomatrix(img, [1], [0], levels=256))

   # print(len(glcmMatrix))
    # get properties
    for j in range(0, len(proList)):
        properties[j]=(greycoprops(glcmMatrix, prop=proList[j]))

    features = np.array([properties[0],properties[1],properties[2],properties[3],properties[4],output_Y[i]])
    #print(features)
    final.append(features)

df = pd.DataFrame(final,columns=featlist)
df.to_excel('/content/drive/My Drive/features.xlsx')
#The data is split into 80% training and 20% testing in this block. These variables will be used for all machine learning models.
X=np.array((df.as_matrix()))
Y=X[:,5]
X=X[:,0:5]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=10)
y_train = y_train.astype('int')
y_test = y_test.astype('int')