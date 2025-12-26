# 87Sr-ion-energy-level
Interactive chart of the energy levels of 87Sr+ ion

Required package: streamlit sympy pandas numpy plotly, **be careful about the version!**

1. pip install -r requirements.txt
2. streamlit run sr87_intermediate_levels_full_line.py

# Brief Manual:
  ## 1. After running the code, a web page will automatically open in the browser.
  Here is a sample of the default main page.
  
  ![page part1](/Figure/intro1.png)
  ![page part2](/Figure/intro2.png)
  ![page part3](/Figure/intro3.png)

  ## 2. There are 7 blocks in the page:
     
  **Block 1**: Biref intro about the program and the formula used in it
     
  **Block 2**: Selected states to shown on the Block 4 main figure
     
  **Block 3**: Two types of magentic field input for calculating the zeeman splitting in Block 4 & 5
     
  **Block 4**: Main figure to show the states
     
  **Block 5**: Calculation of selected two Zeeman sub levels

  **Block 6**: eigenstate expressed in the |F,mF> and |mI,mJ> basis
     
  **Block 7**: Relation of Zeeman splitting in side the selected state with magnetic field

  ### Possible errors: 
  1. Since there are two default levels in Block5, if the related states is deleted in Block 2, there would be error, one solution is changing the defulat levels before delete the state in Block 2, the other solution is delete all states in Block 2 and add the only needed states then calculate.
  2. At the first time changing the B field value in Block 3, there would be some warning message, just ignore it.

  ## 3. Inside the Block 5, there is a fold sub block, it shows the frequency difference with external static magnetic field, and calculate the insentitive point if availible.
  
  **Step 1**: select needed state
  
  **Step 2**: Select levels inside the state
  
  **Step 3**: unfold the sub block 
  ![Transition Sub Block](/Figure/intro4.png)


Running streamlit require an email address.
