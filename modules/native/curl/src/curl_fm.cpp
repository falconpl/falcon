/*
   FALCON - The Falcon Programming Language.
   FILE: curl_fm.cpp

   cURL library binding for Falcon
   Main module class
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

#include "curl_fm.h"
#include "curl_ext.h"
#include <curl/curl.h>

namespace Falcon {
namespace Canonical {

int ModuleCurl::init_count = 0;

ModuleCurl::ModuleCurl():
   Module(FALCON_CANONICAL_CURL_NAME, true)
{
   if( init_count == 0 )
   {
      curl_global_init( CURL_GLOBAL_ALL );
   }

   ++init_count;

   //====================================================

   m_handleClass = new Curl::ClassHandle;
   m_multiClass = new Curl::ClassMulti;

   *this
      << new Curl::FALCON_FUNCTION_NAME(dload)
      << m_handleClass
      << m_multiClass
      << new Curl::ClassCURL
      << new Curl::ClassAUTH
      << new Curl::ClassFTPAUTH
      << new Curl::ClassFTPMETHOD
      << new Curl::ClassHTTP
      << new Curl::ClassINFO
      << new Curl::ClassIPRESOLVE
      << new Curl::ClassNETRC
      << new Curl::ClassOPT
      << new Curl::ClassPROXY
      << new Curl::ClassSSH_AUTH
      << new Curl::ClassSSLVERSION
      << new Curl::ClassSSL_CCC
      << new Curl::ClassUSESSL

      << new Curl::ClassCurlError

      ;
}


ModuleCurl::~ModuleCurl()
{
   if( --init_count == 0 )
      curl_global_cleanup();
}

}
}

/* end of curl_fm.cpp */

