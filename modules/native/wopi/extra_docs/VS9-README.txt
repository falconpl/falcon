                                       
                                       FALCON WOPI 1.0

                                   MS-Windows pre-built files
                   
            
                              For Falcon built with Visual Studio 9
    

This package contains 2 executable programs, falhttpd and falcgi, that are respectively
the Falcon WOPI stand-alone web server and the CGI front-end.

Their usage is explained in the HTML documentation you can find under the docs/ 
directory. The programs doesn't need to be installed, as long as the Falcon libraries
and the Visual Studio Runtime can be readily found (a normal Falcon installation will
ensure that). They can be used directly from any location you launch them.

Also, the cgi_fm.dll file is a Falcon module that should be copied in the Falcon binary
directory (where falcon.exe resides), and that can be loaded by WOPI scripts willing
to be used as CGI applications from web servers.

More details on configuration and usage of this items is explained in the HTML
documentation that you can find under the docs directory.

 


               