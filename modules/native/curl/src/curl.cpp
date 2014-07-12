/*
   FALCON - The Falcon Programming Language.
   FILE: curl_ext.cpp

   cURL library binding for Falcon
   Main module file, providing the module object to
   the Falcon engine.
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
   Main module file, providing the module object to
   the Falcon engine.
*/

#include "curl_fm.h"


/*#
   @module curl cURL Http/Ftp library binding.

   This module provides a tight and complete integration with
   the @link "http://curl.haxx.se/libcurl/" libcurl library.

   Libcurl provides a complete set of RFC Internet protocol
   clients and allows a Falcon program to download remote
   files through simple commands.

   The curl Falcon module is structured in a way that allows
   to handle multiple downloads in a single thread, and even in
   a simple coroutine, simplifying by orders of magnitude the
   complexity of sophisticated client programs.

   @section code_status Status of this binding.

   Currently the @b curl module presents a minimal interface to the
   underlying libCURL. The library is actually served through Falcon-wise
   objects and structures. Some of the most advanced features in the
   library are still not bound, but you'll find everything you need to
   upload or download files, send POST http requests, get transfer information
   and basically manage multiplexed transfers.

   More advance binding is scheduled for the next version of this library,
   that will take advantage of a new binding engine in Falcon 0.9.8.

   @section load_request Importing the curl module.

   Since the names of the classes that are declared in this module
   are short and simple, it is advisable to use the @b import directive
   to store the module in its own namespace. For example:
   @code
      import from curl

      h = curl.Handle()
   @endcode

   @section enums Libcurl enumerations.

   The library wrapped by this module, libcurl, uses various sets of @b define
   directives to specify parameters and configure connection values.

   To reduce the complexity of this module, each set of enumerations is stored
   in a different Falcon enumerative class. For example, all the options
   starting with "CURLOPT_" are stored in the OPT enumeration. The option
   that sets the overall operation timeout for a given curl handle can be set
   through the OPT.TIMEOUT option (which corresponds to the CURLOPT_TIMEOUT
   define in the original C API of libcurl).
*/

FALCON_MODULE_DECL
{
   // initialize the module
   Falcon::Module *self = new ::Falcon::Canonical::ModuleCurl();
   return self;

}

/* end of curl.cpp */
