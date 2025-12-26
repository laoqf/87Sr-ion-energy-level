# 87Sr-ion-energy-level
Interactive chart of the energy levels of 87Sr+ ion

Required package: streamlit sympy pandas numpy plotly, **be careful about the version!**

1. pip install -r requirements.txt
2. streamlit run sr87_intermediate_levels_full_line.py

Running streamlit require an email address.

# Brief Manual:
  ## 1. After running the code, a web page will automatically open in the browser.
  Here is a sample of the default main page.
  
  ![page part1](/Figure/intro1.png)
  ![page part2](/Figure/intro2.png)
  ![page part3](/Figure/intro3.png)

  ## 2. There are 7 blocks in the page:
     
  **Block 1 - Introduction**: Brief intro about the program and the Hamiltonian of Zeeman splitting in Hyperfine Structure
     
  **Block 2 - State Selection**: Selected states to be shown on the Block 4
     
  **Block 3 - Sttaic Magnetic Field**: Provide two types of magentic field input for calculating the zeeman splitting in Block 4 & 5
     
  **Block 4 - Main Figure**: Display the selected atomic states and their Zeeman sub levels
     
  **Block 5 - Transition Calculation**: Transition frequency calculation of selected two Zeeman sub levels

  **Block 6 - Eigenstate Expression**: Eigenstate expressed in the |F,mF> and |mI,mJ> basis, if B != 0, Eigenstate would be superposition state of those basis.
     
  **Block 7 - Energy vs B field**: Relation of Zeeman splitting with magnetic field

  ### Possible errors: 
  1. Since there are two default levels in Block5, if the related states is deleted in Block 2, there would be error, one solution is changing the defulat levels before delete the state in Block 2, the other solution is delete all states in Block 2 and add the only needed states then calculate.
  2. At the first time changing the B field value in Block 3, there would be some warning message, just ignore it.

  ## 3. Inside the Block 5, there is a fold sub block, it shows the frequency difference with external static magnetic field, and calculate the insentitive point if availible.
  
  **Step 1**: select needed state
  
  **Step 2**: Select levels inside the state
  
  **Step 3**: unfold the sub block 
  ![Transition Sub Block](/Figure/intro4.png)



