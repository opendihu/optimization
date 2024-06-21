
    #include <iostream>
    #include <cstdlib>
    #include <fstream>
    #include <precice/precice.hpp>
    
    int main()
    {
      std::ofstream file("install_precice-config.xml");
      file << R"(<?xml version="1.0"?>
<precice-configuration>
    
    <data:scalar name="Data"/>
    <mesh name="Mesh1" dimensions="3">
      <use-data name="Data"/>
    </mesh>
    <mesh name="Mesh2" dimensions="3">
      <use-data name="Data"/>
    </mesh>
    
    <participant name="Participant1">
      <provide-mesh name="Mesh1"/>
      <write-data name="Data" mesh="Mesh1"/>    
    </participant>

    <participant name="Participant2">
      <receive-mesh name="Mesh1" from="Participant1"/>
      <provide-mesh name="Mesh2"/>
      <read-data name="Data" mesh="Mesh2"/>
      <mapping:nearest-neighbor
        direction="read"
        from="Mesh1"
        to="Mesh2"
        constraint="consistent" />
    </participant>
    
    <m2n:sockets acceptor="Participant1" connector="Participant2" network="lo" />
    <coupling-scheme:serial-explicit>
      <participants first="Participant1" second="Participant2"/>
      <time-window-size value="0.01"/>
      <max-time value="0.05"/>
      <exchange data="Data" mesh="Mesh1" from="Participant1" to="Participant2"/>
    </coupling-scheme:serial-explicit>

</precice-configuration>
)";
      file.close();
    
      precice::Participant participant("Participant1","install_precice-config.xml",0,1);
      //participant.initialize();
      //participant.finalize();
      
      int ret = system("rm -f install_precice-config.xml");
    
      return EXIT_SUCCESS;
    }

