
      #include <stdlib.h>
      #include <stdio.h>
      #include <iostream>
      #include <sstream>
      #include "opencor.h"
      
      int main(int argc, char* argv[])
      {
        std::stringstream command;
        command << OPENCOR_BINARY << " --version";
        int ret = system(command.str().c_str());
        if (ret == 0)
        {
          std::cout << "opencor found.";
          return EXIT_SUCCESS;
        }
        else
        {
          std::cout << "opencor not found";
          return EXIT_FAILURE;
        }
      }
    
