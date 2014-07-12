/*
   FALCON - The Falcon Programming Language.
   FILE: curl_ext.cpp

   cURL library binding for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 27 Nov 2009 16:31:15 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: The above AUTHOR

         Licensed under the Falcon Programming Language License,
      Version 1.1 (the "License"); you may not use this file
      except in compliance with the License. You may obtain
      a copy of the License at

         http://www.falconpl.org/?page_id=license_1_1

      Unless required by applicable law or agreed to in writing,
      software distributed under the License is distributed on
      an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
      KIND, either express or implied. See the License for the
      specific language governing permissions and limitations
      under the License.

*/

/** \file
   cURL library binding for Falcon
   Interface extension functions - header file
*/

#ifndef FALCON_MODULE_CURL_EXT_H
#define FALCON_MODULE_CURL_EXT_H

#include <falcon/class.h>
#include <falcon/function.h>
#include <falcon/error.h>

namespace Falcon {
class PStep;

namespace Curl {

FALCON_DECLARE_FUNCTION(dload, "uri:S|Uri,stream:[Stream]");

class ClassHandle: public ::Falcon::Class
{
public:
   ClassHandle();
   virtual ~ClassHandle();

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   const PStep* afterExec() const { return m_afterExec; }

private:
   PStep* m_afterExec;
};


class ClassMulti: public ::Falcon::Class
{
public:
   ClassMulti();
   virtual ~ClassMulti();

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
};


class ClassCURL: public ::Falcon::Class
{
public:
   ClassCURL();
   virtual ~ClassCURL();

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;
};


class ClassOPT: public ::Falcon::Class
{
public:
   ClassOPT();
   virtual ~ClassOPT() {}

   virtual void dispose( void* ) const {}
   virtual void* clone( void* ) const {return 0;}
   virtual void* createInstance() const {return 0;}
};


class ClassINFO: public ::Falcon::Class
{
public:
   ClassINFO();
   virtual ~ClassINFO() {}

   virtual void dispose( void* ) const {}
   virtual void* clone( void* ) const {return 0;}
   virtual void* createInstance() const {return 0;}
};


class ClassPROXY: public ::Falcon::Class
{
public:
   ClassPROXY();
   virtual ~ClassPROXY() {}

   virtual void dispose( void* ) const {}
   virtual void* clone( void* ) const {return 0;}
   virtual void* createInstance() const {return 0;}
};

class ClassNETRC: public ::Falcon::Class
{
public:
   ClassNETRC();
   virtual ~ClassNETRC() {}

   virtual void dispose( void* ) const {}
   virtual void* clone( void* ) const {return 0;}
   virtual void* createInstance() const {return 0;}
};

class ClassAUTH: public ::Falcon::Class
{
public:
   ClassAUTH();
   virtual ~ClassAUTH() {}

   virtual void dispose( void* ) const {}
   virtual void* clone( void* ) const {return 0;}
   virtual void* createInstance() const {return 0;}
};

class ClassHTTP: public ::Falcon::Class
{
public:
   ClassHTTP();
   virtual ~ClassHTTP() {}

   virtual void dispose( void* ) const {}
   virtual void* clone( void* ) const {return 0;}
   virtual void* createInstance() const {return 0;}
};

class ClassUSESSL: public ::Falcon::Class
{
public:
   ClassUSESSL();
   virtual ~ClassUSESSL() {}

   virtual void dispose( void* ) const {}
   virtual void* clone( void* ) const {return 0;}
   virtual void* createInstance() const {return 0;}
};


class ClassFTPAUTH: public ::Falcon::Class
{
public:
   ClassFTPAUTH();
   virtual ~ClassFTPAUTH() {}

   virtual void dispose( void* ) const {}
   virtual void* clone( void* ) const {return 0;}
   virtual void* createInstance() const {return 0;}
};


class ClassSSL_CCC: public ::Falcon::Class
{
public:
   ClassSSL_CCC();
   virtual ~ClassSSL_CCC() {}

   virtual void dispose( void* ) const {}
   virtual void* clone( void* ) const {return 0;}
   virtual void* createInstance() const {return 0;}
};


class ClassFTPMETHOD: public ::Falcon::Class
{
public:
   ClassFTPMETHOD();
   virtual ~ClassFTPMETHOD() {}

   virtual void dispose( void* ) const {}
   virtual void* clone( void* ) const {return 0;}
   virtual void* createInstance() const {return 0;}
};


class ClassIPRESOLVE: public ::Falcon::Class
{
public:
   ClassIPRESOLVE();
   virtual ~ClassIPRESOLVE() {}

   virtual void dispose( void* ) const {}
   virtual void* clone( void* ) const {return 0;}
   virtual void* createInstance() const {return 0;}
};


class ClassSSLVERSION: public ::Falcon::Class
{
public:
   ClassSSLVERSION();
   virtual ~ClassSSLVERSION() {}

   virtual void dispose( void* ) const {}
   virtual void* clone( void* ) const {return 0;}
   virtual void* createInstance() const {return 0;}
};


class ClassSSH_AUTH: public ::Falcon::Class
{
public:
   ClassSSH_AUTH();
   virtual ~ClassSSH_AUTH() {}

   virtual void dispose( void* ) const {}
   virtual void* clone( void* ) const {return 0;}
   virtual void* createInstance() const {return 0;}
};


FALCON_DECLARE_ERROR( CurlError );

}
}

#endif

/* end of curl_ext.h */
