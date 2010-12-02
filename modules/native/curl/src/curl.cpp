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

#include <curl/curl.h>
#include <falcon/module.h>
#include "curl_ext.h"
#include "curl_mod.h"
#include "curl_st.h"

#include "version.h"

//==================================================
// Extension of Falcon module
//==================================================

class CurlModule: public Falcon::Module
{
   static int init_count;

public:
   CurlModule();
   virtual ~CurlModule();
};


int CurlModule::init_count = 0;

CurlModule::CurlModule():
   Module()
{
   if( init_count == 0 )
   {
      curl_global_init( CURL_GLOBAL_ALL );
   }

   ++init_count;
}


CurlModule::~CurlModule()
{
   if( --init_count == 0 )
      curl_global_cleanup();
}

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
   #define FALCON_DECLARE_MODULE self

   // initialize the module
   Falcon::Module *self = new CurlModule();
   self->name( "curl" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //============================================================
   // Here declare the international string table implementation
   //
   #include "curl_st.h"

   //============================================================
   // Here declare CURL - easy api
   //
   self->addExtFunc( "curl_version", Falcon::Ext::curl_version );
   self->addExtFunc( "dload", Falcon::Ext::curl_dload )->
         addParam("uri")->addParam("stream");

   Falcon::Symbol *easy_class = self->addClass( "Handle", Falcon::Ext::Handle_init )
      ->addParam( "uri" );
   easy_class->setWKS(true);
   easy_class->getClassDef()->factory( &Falcon::Mod::CurlHandle::Factory );
   self->addClassMethod( easy_class, "exec", Falcon::Ext::Handle_exec );

   self->addClassMethod( easy_class, "setOutConsole", &Falcon::Ext::Handle_setOutConsole );
   self->addClassMethod( easy_class, "setOutString", &Falcon::Ext::Handle_setOutString );
   self->addClassMethod( easy_class, "setOutStream", &Falcon::Ext::Handle_setOutStream ).asSymbol()
      ->addParam( "stream" );
   self->addClassMethod( easy_class, "setOutCallback", &Falcon::Ext::Handle_setOutCallback ).asSymbol()
      ->addParam( "cb" );
   self->addClassMethod( easy_class, "setInStream", &Falcon::Ext::Handle_setInStream ).asSymbol()
      ->addParam( "stream" );
   self->addClassMethod( easy_class, "setInCallback", &Falcon::Ext::Handle_setInCallback ).asSymbol()
      ->addParam( "cb" );

   //self->addClassMethod( easy_class, "setOutMessage", Falcon::Ext::Handle_setOutMessage ).asSymbol()
   //   ->addParam( "slot" );
   self->addClassMethod( easy_class, "getData", &Falcon::Ext::Handle_getData );
   self->addClassMethod( easy_class, "setOption", &Falcon::Ext::Handle_setOption ).asSymbol()
      ->addParam( "option" )->addParam( "data" );
   self->addClassMethod( easy_class, "setOptions", &Falcon::Ext::Handle_setOptions ).asSymbol()
      ->addParam( "options" );
   self->addClassMethod( easy_class, "postData", &Falcon::Ext::Handle_postData ).asSymbol()
      ->addParam( "data" );
   self->addClassMethod( easy_class, "getInfo", &Falcon::Ext::Handle_getInfo ).asSymbol()
      ->addParam( "option" );

   self->addClassMethod( easy_class, "cleanup", &Falcon::Ext::Handle_cleanup );
   self->addClassProperty( easy_class, "data" );

   //============================================================
   // Here declare CURL - multi api
   //

   Falcon::Symbol *multy_class = self->addClass( "Multi", &Falcon::Ext::Multi_init );
   multy_class->getClassDef()->factory( &Falcon::Mod::CurlMultiHandle::Factory );
   self->addClassMethod( multy_class, "add", &Falcon::Ext::Multi_add ).asSymbol()
      ->addParam( "h" );
   self->addClassMethod( multy_class, "remove", &Falcon::Ext::Multi_remove ).asSymbol()
      ->addParam( "h" );
   self->addClassMethod( multy_class, "perform", &Falcon::Ext::Multi_perform );


   //============================================================
   // Enumeration class CURL
   //
   Falcon::Symbol *curle_class = self->addClass( "CURL" );
   self->addClassProperty( curle_class, "READFUNC_ABORT" ).setInteger(CURL_READFUNC_ABORT).setReadOnly(true);
   self->addClassProperty( curle_class, "READFUNC_PAUSE" ).setInteger(CURL_READFUNC_PAUSE).setReadOnly(true);
   self->addClassProperty( curle_class, "WRITEFUNC_PAUSE" ).setInteger(CURL_WRITEFUNC_PAUSE).setReadOnly(true);

   //============================================================
   // Enumeration class COPT
   //
   Falcon::Symbol *copt_class = self->addClass( "OPT" );
   self->addClassProperty( copt_class, "VERBOSE" ).setInteger(CURLOPT_VERBOSE).setReadOnly(true);
   self->addClassProperty( copt_class, "HEADER" ).setInteger(CURLOPT_HEADER).setReadOnly(true);
   self->addClassProperty( copt_class, "NOPROGRESS" ).setInteger(CURLOPT_NOPROGRESS).setReadOnly(true);

   self->addClassProperty( copt_class, "HTTPPROXYTUNNEL" ).setInteger(CURLOPT_HTTPPROXYTUNNEL).setReadOnly(true);
#if LIBCURL_VERSION_NUM >= 0x071904
   self->addClassProperty( copt_class, "SOCKS5_GSSAPI_NEC" ).setInteger(CURLOPT_SOCKS5_GSSAPI_NEC).setReadOnly(true);
#endif
   self->addClassProperty( copt_class, "TCP_NODELAY" ).setInteger(CURLOPT_TCP_NODELAY).setReadOnly(true);
   self->addClassProperty( copt_class, "AUTOREFERER" ).setInteger(CURLOPT_AUTOREFERER).setReadOnly(true);
   self->addClassProperty( copt_class, "FOLLOWLOCATION" ).setInteger(CURLOPT_FOLLOWLOCATION).setReadOnly(true);
   self->addClassProperty( copt_class, "UNRESTRICTED_AUTH" ).setInteger(CURLOPT_UNRESTRICTED_AUTH).setReadOnly(true);
   self->addClassProperty( copt_class, "PUT" ).setInteger(CURLOPT_PUT).setReadOnly(true);
   self->addClassProperty( copt_class, "POST" ).setInteger(CURLOPT_POST).setReadOnly(true);
   self->addClassProperty( copt_class, "COOKIESESSION" ).setInteger(CURLOPT_COOKIESESSION).setReadOnly(true);
   self->addClassProperty( copt_class, "HTTPGET" ).setInteger(CURLOPT_HTTPGET).setReadOnly(true);
   self->addClassProperty( copt_class, "IGNORE_CONTENT_LENGTH" ).setInteger(CURLOPT_IGNORE_CONTENT_LENGTH).setReadOnly(true);
   self->addClassProperty( copt_class, "DIRLISTONLY" ).setInteger(CURLOPT_DIRLISTONLY).setReadOnly(true);
   self->addClassProperty( copt_class, "APPEND" ).setInteger(CURLOPT_APPEND).setReadOnly(true);
   self->addClassProperty( copt_class, "FTP_USE_EPRT" ).setInteger(CURLOPT_FTP_USE_EPRT).setReadOnly(true);
   self->addClassProperty( copt_class, "FTP_USE_EPSV" ).setInteger(CURLOPT_FTP_USE_EPSV).setReadOnly(true);
   self->addClassProperty( copt_class, "FTP_CREATE_MISSING_DIRS" ).setInteger(CURLOPT_FTP_CREATE_MISSING_DIRS).setReadOnly(true);
   self->addClassProperty( copt_class, "FTP_SKIP_PASV_IP" ).setInteger(CURLOPT_FTP_SKIP_PASV_IP).setReadOnly(true);
   self->addClassProperty( copt_class, "TRANSFERTEXT" ).setInteger(CURLOPT_TRANSFERTEXT).setReadOnly(true);
   self->addClassProperty( copt_class, "PROXY_TRANSFER_MODE" ).setInteger(CURLOPT_PROXY_TRANSFER_MODE).setReadOnly(true);
   self->addClassProperty( copt_class, "CRLF" ).setInteger(CURLOPT_CRLF).setReadOnly(true);
   self->addClassProperty( copt_class, "FILETIME" ).setInteger(CURLOPT_FILETIME).setReadOnly(true);
   self->addClassProperty( copt_class, "NOBODY" ).setInteger(CURLOPT_NOBODY).setReadOnly(true);
   self->addClassProperty( copt_class, "UPLOAD" ).setInteger(CURLOPT_UPLOAD).setReadOnly(true);
   self->addClassProperty( copt_class, "FRESH_CONNECT" ).setInteger(CURLOPT_FRESH_CONNECT).setReadOnly(true);
   self->addClassProperty( copt_class, "FORBID_REUSE" ).setInteger(CURLOPT_FORBID_REUSE).setReadOnly(true);
   self->addClassProperty( copt_class, "CONNECT_ONLY" ).setInteger(CURLOPT_CONNECT_ONLY).setReadOnly(true);
   self->addClassProperty( copt_class, "SSLENGINE_DEFAULT" ).setInteger(CURLOPT_SSLENGINE_DEFAULT).setReadOnly(true);
   self->addClassProperty( copt_class, "SSL_VERIFYPEER" ).setInteger(CURLOPT_SSL_VERIFYPEER).setReadOnly(true);
#if LIBCURL_VERSION_NUM >= 0x071901
   self->addClassProperty( copt_class, "CERTINFO" ).setInteger(CURLOPT_CERTINFO).setReadOnly(true);
#endif
   self->addClassProperty( copt_class, "SSL_VERIFYHOST" ).setInteger(CURLOPT_SSL_VERIFYHOST).setReadOnly(true);
   self->addClassProperty( copt_class, "SSL_SESSIONID_CACHE" ).setInteger(CURLOPT_SSL_SESSIONID_CACHE).setReadOnly(true);

   self->addClassProperty( copt_class, "PROTOCOLS" ).setInteger(CURLOPT_PROTOCOLS).setReadOnly(true);
   self->addClassProperty( copt_class, "REDIR_PROTOCOLS" ).setInteger(CURLOPT_REDIR_PROTOCOLS).setReadOnly(true);
   self->addClassProperty( copt_class, "PROXYPORT" ).setInteger(CURLOPT_PROXYPORT).setReadOnly(true);
   self->addClassProperty( copt_class, "PROXYTYPE" ).setInteger(CURLOPT_PROXYTYPE).setReadOnly(true);


   Falcon::Symbol *cproxy_class = self->addClass( "PROXY" );
   self->addClassProperty( cproxy_class, "HTTP" ).setInteger(CURLPROXY_HTTP).setReadOnly(true);
#if LIBCURL_VERSION_NUM >= 0x071904
   self->addClassProperty( cproxy_class, "HTTP_1_0" ).setInteger(CURLPROXY_HTTP_1_0).setReadOnly(true);
#endif
   self->addClassProperty( cproxy_class, "SOCKS4" ).setInteger(CURLPROXY_SOCKS4).setReadOnly(true);
   self->addClassProperty( cproxy_class, "SOCKS5" ).setInteger(CURLPROXY_SOCKS5).setReadOnly(true);
   self->addClassProperty( cproxy_class, "SOCKS4A" ).setInteger(CURLPROXY_SOCKS4A).setReadOnly(true);

   self->addClassProperty( copt_class, "LOCALPORT" ).setInteger(CURLOPT_LOCALPORT).setReadOnly(true);
   self->addClassProperty( copt_class, "LOCALPORTRANGE" ).setInteger(CURLOPT_LOCALPORTRANGE).setReadOnly(true);
   self->addClassProperty( copt_class, "DNS_CACHE_TIMEOUT" ).setInteger(CURLOPT_DNS_CACHE_TIMEOUT).setReadOnly(true);
   self->addClassProperty( copt_class, "DNS_USE_GLOBAL_CACHE" ).setInteger(CURLOPT_DNS_USE_GLOBAL_CACHE).setReadOnly(true);
   self->addClassProperty( copt_class, "BUFFERSIZE" ).setInteger(CURLOPT_BUFFERSIZE).setReadOnly(true);
   self->addClassProperty( copt_class, "PORT" ).setInteger(CURLOPT_PORT).setReadOnly(true);
#if LIBCURL_VERSION_NUM >= 0x071900
   self->addClassProperty( copt_class, "ADDRESS_SCOPE" ).setInteger(CURLOPT_ADDRESS_SCOPE).setReadOnly(true);
#endif
   self->addClassProperty( copt_class, "NETRC" ).setInteger(CURLOPT_NETRC).setReadOnly(true);


   Falcon::Symbol *cnetrc_class = self->addClass( "NETRC" );
   self->addClassProperty( cnetrc_class, "OPTIONAL" ).setInteger(CURL_NETRC_OPTIONAL).setReadOnly(true);
   self->addClassProperty( cnetrc_class, "IGNORED" ).setInteger(CURL_NETRC_IGNORED).setReadOnly(true);

   self->addClassProperty( copt_class, "HTTPAUTH" ).setInteger(CURLOPT_HTTPAUTH).setReadOnly(true);

   Falcon::Symbol *cauth_class = self->addClass( "AUTH" );
   self->addClassProperty( cauth_class, "BASIC" ).setInteger(CURLAUTH_BASIC).setReadOnly(true);
   self->addClassProperty( cauth_class, "DIGEST" ).setInteger(CURLAUTH_DIGEST).setReadOnly(true);
#if LIBCURL_VERSION_NUM >= 0x071903
   self->addClassProperty( cauth_class, "DIGEST_IE" ).setInteger(CURLAUTH_DIGEST_IE).setReadOnly(true);
#endif
   self->addClassProperty( cauth_class, "GSSNEGOTIATE" ).setInteger(CURLAUTH_GSSNEGOTIATE).setReadOnly(true);
   self->addClassProperty( cauth_class, "NTLM" ).setInteger(CURLAUTH_NTLM).setReadOnly(true);
   self->addClassProperty( cauth_class, "ANY" ).setInteger(CURLAUTH_ANY).setReadOnly(true);
   self->addClassProperty( cauth_class, "ANYSAFE" ).setInteger(CURLAUTH_ANYSAFE).setReadOnly(true);

   self->addClassProperty( copt_class, "PROXYAUTH" ).setInteger(CURLOPT_PROXYAUTH).setReadOnly(true);
   self->addClassProperty( copt_class, "MAXREDIRS" ).setInteger(CURLOPT_MAXREDIRS).setReadOnly(true);
#if LIBCURL_VERSION_NUM >= 0x071901
   self->addClassProperty( copt_class, "POSTREDIR" ).setInteger(CURLOPT_POSTREDIR).setReadOnly(true);
#endif
   self->addClassProperty( copt_class, "HTTP_VERSION" ).setInteger(CURLOPT_HTTP_VERSION).setReadOnly(true);

   Falcon::Symbol *chttp_class = self->addClass( "HTTP" );
   self->addClassProperty( chttp_class, "VERSION_NONE" ).setInteger(CURL_HTTP_VERSION_NONE).setReadOnly(true);
   self->addClassProperty( chttp_class, "VERSION_1_0" ).setInteger(CURL_HTTP_VERSION_1_0).setReadOnly(true);
   self->addClassProperty( chttp_class, "VERSION_1_1" ).setInteger(CURL_HTTP_VERSION_1_1).setReadOnly(true);

   self->addClassProperty( copt_class, "HTTP_CONTENT_DECODING" ).setInteger(CURLOPT_HTTP_CONTENT_DECODING).setReadOnly(true);
   self->addClassProperty( copt_class, "HTTP_TRANSFER_DECODING" ).setInteger(CURLOPT_HTTP_TRANSFER_DECODING).setReadOnly(true);
#if LIBCURL_VERSION_NUM >= 0x071904
   self->addClassProperty( copt_class, "TFTP_BLKSIZE" ).setInteger(CURLOPT_TFTP_BLKSIZE).setReadOnly(true);
#endif
   self->addClassProperty( copt_class, "FTP_RESPONSE_TIMEOUT" ).setInteger(CURLOPT_FTP_RESPONSE_TIMEOUT).setReadOnly(true);
   self->addClassProperty( copt_class, "USE_SSL" ).setInteger(CURLOPT_USE_SSL).setReadOnly(true);

   Falcon::Symbol *cusessl_class = self->addClass( "USESSL" );
   self->addClassProperty( cusessl_class, "NONE" ).setInteger(CURLUSESSL_NONE).setReadOnly(true);
   self->addClassProperty( cusessl_class, "TRY" ).setInteger(CURLUSESSL_TRY).setReadOnly(true);
   self->addClassProperty( cusessl_class, "CONTROL" ).setInteger(CURLUSESSL_CONTROL).setReadOnly(true);
   self->addClassProperty( cusessl_class, "ALL" ).setInteger(CURLUSESSL_ALL).setReadOnly(true);

   self->addClassProperty( copt_class, "FTPSSLAUTH" ).setInteger(CURLOPT_FTPSSLAUTH).setReadOnly(true);

   Falcon::Symbol *cftpauth_class = self->addClass( "FTPAUTH" );
   self->addClassProperty( cftpauth_class, "DEFAULT" ).setInteger(CURLFTPAUTH_DEFAULT).setReadOnly(true);
   self->addClassProperty( cftpauth_class, "SSL" ).setInteger(CURLFTPAUTH_SSL).setReadOnly(true);
   self->addClassProperty( cftpauth_class, "TLS" ).setInteger(CURLFTPAUTH_TLS).setReadOnly(true);

   self->addClassProperty( copt_class, "FTP_SSL_CCC" ).setInteger(CURLOPT_FTP_SSL_CCC).setReadOnly(true);

   Falcon::Symbol *cftpssl_ccc_class = self->addClass( "FTPSSL_CCC" );
   self->addClassProperty( cftpssl_ccc_class, "NONE" ).setInteger(CURLFTPSSL_CCC_NONE).setReadOnly(true);
   self->addClassProperty( cftpssl_ccc_class, "PASSIVE" ).setInteger(CURLFTPSSL_CCC_PASSIVE).setReadOnly(true);
   self->addClassProperty( cftpssl_ccc_class, "ACTIVE" ).setInteger(CURLFTPSSL_CCC_ACTIVE).setReadOnly(true);


   self->addClassProperty( copt_class, "FTP_FILEMETHOD" ).setInteger(CURLOPT_FTP_FILEMETHOD).setReadOnly(true);

   Falcon::Symbol *cftpmethod_class = self->addClass( "FTPMETHOD" );
   self->addClassProperty( cftpmethod_class, "MULTICWD" ).setInteger(CURLFTPMETHOD_MULTICWD).setReadOnly(true);
   self->addClassProperty( cftpmethod_class, "NOCWD" ).setInteger(CURLFTPSSL_CCC_PASSIVE).setReadOnly(true);
   self->addClassProperty( cftpmethod_class, "SINGLECWD" ).setInteger(CURLFTPMETHOD_SINGLECWD).setReadOnly(true);

   self->addClassProperty( copt_class, "RESUME_FROM" ).setInteger(CURLOPT_RESUME_FROM).setReadOnly(true);
   self->addClassProperty( copt_class, "INFILESIZE" ).setInteger(CURLOPT_INFILESIZE).setReadOnly(true);
   self->addClassProperty( copt_class, "MAXFILESIZE" ).setInteger(CURLOPT_MAXFILESIZE).setReadOnly(true);
   self->addClassProperty( copt_class, "TIMEVALUE" ).setInteger(CURLOPT_TIMEVALUE).setReadOnly(true);
   self->addClassProperty( copt_class, "TIMEOUT" ).setInteger(CURLOPT_TIMEOUT).setReadOnly(true);
   self->addClassProperty( copt_class, "TIMEOUT_MS" ).setInteger(CURLOPT_TIMEOUT_MS).setReadOnly(true);
   self->addClassProperty( copt_class, "LOW_SPEED_LIMIT" ).setInteger(CURLOPT_LOW_SPEED_LIMIT).setReadOnly(true);
   self->addClassProperty( copt_class, "LOW_SPEED_TIME" ).setInteger(CURLOPT_LOW_SPEED_TIME).setReadOnly(true);
   self->addClassProperty( copt_class, "MAXCONNECTS" ).setInteger(CURLOPT_MAXCONNECTS).setReadOnly(true);
   self->addClassProperty( copt_class, "CONNECTTIMEOUT" ).setInteger(CURLOPT_CONNECTTIMEOUT).setReadOnly(true);
   self->addClassProperty( copt_class, "CONNECTTIMEOUT_MS" ).setInteger(CURLOPT_CONNECTTIMEOUT_MS).setReadOnly(true);
   self->addClassProperty( copt_class, "IPRESOLVE" ).setInteger(CURLOPT_IPRESOLVE).setReadOnly(true);

   Falcon::Symbol *cipresolve_class = self->addClass( "IPRESOLVE" );
   self->addClassProperty( cipresolve_class, "WHATEVER" ).setInteger(CURL_IPRESOLVE_WHATEVER).setReadOnly(true);
   self->addClassProperty( cipresolve_class, "V4" ).setInteger(CURL_IPRESOLVE_V4).setReadOnly(true);
   self->addClassProperty( cipresolve_class, "V6" ).setInteger(CURL_IPRESOLVE_V6).setReadOnly(true);


   self->addClassProperty( copt_class, "SSLVERSION" ).setInteger(CURLOPT_SSLVERSION).setReadOnly(true);

   Falcon::Symbol *csslversion_class = self->addClass( "SSLVERSION" );
   self->addClassProperty( csslversion_class, "DEFAULT" ).setInteger(CURL_SSLVERSION_DEFAULT).setReadOnly(true);
   self->addClassProperty( csslversion_class, "TLSv1" ).setInteger(CURL_SSLVERSION_TLSv1).setReadOnly(true);
   self->addClassProperty( csslversion_class, "SSLv2" ).setInteger(CURL_SSLVERSION_SSLv2).setReadOnly(true);
   self->addClassProperty( csslversion_class, "SSLv3" ).setInteger(CURL_SSLVERSION_SSLv3).setReadOnly(true);

   self->addClassProperty( copt_class, "SSH_AUTH_TYPES" ).setInteger(CURLOPT_SSH_AUTH_TYPES).setReadOnly(true);

   Falcon::Symbol *cssh_auth_class = self->addClass( "SSH_AUTH" );
   self->addClassProperty( cssh_auth_class, "PUBLICKEY" ).setInteger(CURLSSH_AUTH_PUBLICKEY).setReadOnly(true);
   self->addClassProperty( cssh_auth_class, "PASSWORD" ).setInteger(CURLSSH_AUTH_PASSWORD).setReadOnly(true);
   self->addClassProperty( cssh_auth_class, "HOST" ).setInteger(CURLSSH_AUTH_HOST).setReadOnly(true);
   self->addClassProperty( cssh_auth_class, "KEYBOARD" ).setInteger(CURLSSH_AUTH_KEYBOARD).setReadOnly(true);
   self->addClassProperty( cssh_auth_class, "ANY" ).setInteger(CURLSSH_AUTH_ANY).setReadOnly(true);


   self->addClassProperty( copt_class, "NEW_FILE_PERMS" ).setInteger(CURLOPT_NEW_FILE_PERMS).setReadOnly(true);
   self->addClassProperty( copt_class, "NEW_DIRECTORY_PERMS" ).setInteger(CURLOPT_NEW_DIRECTORY_PERMS).setReadOnly(true);

   self->addClassProperty( copt_class, "RESUME_FROM_LARGE" ).setInteger(CURLOPT_RESUME_FROM_LARGE).setReadOnly(true);
   self->addClassProperty( copt_class, "INFILESIZE_LARGE" ).setInteger(CURLOPT_INFILESIZE_LARGE).setReadOnly(true);
   self->addClassProperty( copt_class, "MAXFILESIZE_LARGE" ).setInteger(CURLOPT_MAXFILESIZE_LARGE).setReadOnly(true);
   self->addClassProperty( copt_class, "MAX_SEND_SPEED_LARGE" ).setInteger(CURLOPT_MAX_SEND_SPEED_LARGE).setReadOnly(true);
   self->addClassProperty( copt_class, "MAX_RECV_SPEED_LARGE" ).setInteger(CURLOPT_MAX_RECV_SPEED_LARGE).setReadOnly(true);

   self->addClassProperty( copt_class, "URL" ).setInteger(CURLOPT_URL).setReadOnly(true);
   self->addClassProperty( copt_class, "PROXY" ).setInteger(CURLOPT_PROXY).setReadOnly(true);
#if LIBCURL_VERSION_NUM >= 0x071904
   self->addClassProperty( copt_class, "NOPROXY" ).setInteger(CURLOPT_NOPROXY).setReadOnly(true);
   self->addClassProperty( copt_class, "SOCKS5_GSSAPI_SERVICE" ).setInteger(CURLOPT_SOCKS5_GSSAPI_SERVICE).setReadOnly(true);
#endif
   self->addClassProperty( copt_class, "INTERFACE" ).setInteger(CURLOPT_INTERFACE).setReadOnly(true);
   self->addClassProperty( copt_class, "NETRC_FILE" ).setInteger(CURLOPT_NETRC_FILE).setReadOnly(true);
   self->addClassProperty( copt_class, "USERPWD" ).setInteger(CURLOPT_USERPWD).setReadOnly(true);
   self->addClassProperty( copt_class, "PROXYUSERPWD" ).setInteger(CURLOPT_PROXYUSERPWD).setReadOnly(true);
#if LIBCURL_VERSION_NUM >= 0x071901
   self->addClassProperty( copt_class, "USERNAME" ).setInteger(CURLOPT_USERNAME).setReadOnly(true);
   self->addClassProperty( copt_class, "PASSWORD" ).setInteger(CURLOPT_PASSWORD).setReadOnly(true);
   self->addClassProperty( copt_class, "PROXYUSERNAME" ).setInteger(CURLOPT_PROXYUSERNAME).setReadOnly(true);
   self->addClassProperty( copt_class, "PROXYPASSWORD" ).setInteger(CURLOPT_PROXYPASSWORD).setReadOnly(true);
#endif
   self->addClassProperty( copt_class, "ENCODING" ).setInteger(CURLOPT_ENCODING).setReadOnly(true);
   self->addClassProperty( copt_class, "REFERER" ).setInteger(CURLOPT_REFERER).setReadOnly(true);
   self->addClassProperty( copt_class, "USERAGENT" ).setInteger(CURLOPT_USERAGENT).setReadOnly(true);
   self->addClassProperty( copt_class, "COOKIE" ).setInteger(CURLOPT_COOKIE).setReadOnly(true);
   self->addClassProperty( copt_class, "COOKIEFILE" ).setInteger(CURLOPT_COOKIEFILE).setReadOnly(true);
   self->addClassProperty( copt_class, "COOKIEJAR" ).setInteger(CURLOPT_COOKIEJAR).setReadOnly(true);
   self->addClassProperty( copt_class, "COOKIELIST" ).setInteger(CURLOPT_COOKIELIST).setReadOnly(true);
   self->addClassProperty( copt_class, "FTPPORT" ).setInteger(CURLOPT_FTPPORT).setReadOnly(true);
   self->addClassProperty( copt_class, "FTP_ALTERNATIVE_TO_USER" ).setInteger(CURLOPT_FTP_ALTERNATIVE_TO_USER).setReadOnly(true);
   self->addClassProperty( copt_class, "FTP_ACCOUNT" ).setInteger(CURLOPT_FTP_ACCOUNT).setReadOnly(true);
   self->addClassProperty( copt_class, "RANGE" ).setInteger(CURLOPT_RANGE).setReadOnly(true);
   self->addClassProperty( copt_class, "CUSTOMREQUEST" ).setInteger(CURLOPT_CUSTOMREQUEST).setReadOnly(true);
   self->addClassProperty( copt_class, "SSLCERT" ).setInteger(CURLOPT_SSLCERT).setReadOnly(true);
   self->addClassProperty( copt_class, "SSLCERTTYPE" ).setInteger(CURLOPT_SSLCERTTYPE).setReadOnly(true);
   self->addClassProperty( copt_class, "SSLKEY" ).setInteger(CURLOPT_SSLKEY).setReadOnly(true);
   self->addClassProperty( copt_class, "SSLKEYTYPE" ).setInteger(CURLOPT_SSLKEYTYPE).setReadOnly(true);
   self->addClassProperty( copt_class, "KEYPASSWD" ).setInteger(CURLOPT_KEYPASSWD).setReadOnly(true);
   self->addClassProperty( copt_class, "SSLENGINE" ).setInteger(CURLOPT_SSLENGINE).setReadOnly(true);
   self->addClassProperty( copt_class, "CAINFO" ).setInteger(CURLOPT_CAINFO).setReadOnly(true);
#if LIBCURL_VERSION_NUM >= 0x071900
   self->addClassProperty( copt_class, "ISSUERCERT" ).setInteger(CURLOPT_ISSUERCERT).setReadOnly(true);
   self->addClassProperty( copt_class, "CRLFILE" ).setInteger(CURLOPT_CRLFILE).setReadOnly(true);
#endif
   self->addClassProperty( copt_class, "CAPATH" ).setInteger(CURLOPT_CAPATH).setReadOnly(true);
   self->addClassProperty( copt_class, "RANDOM_FILE" ).setInteger(CURLOPT_RANDOM_FILE).setReadOnly(true);
   self->addClassProperty( copt_class, "EGDSOCKET" ).setInteger(CURLOPT_EGDSOCKET).setReadOnly(true);
   self->addClassProperty( copt_class, "SSL_CIPHER_LIST" ).setInteger(CURLOPT_SSL_CIPHER_LIST).setReadOnly(true);
   self->addClassProperty( copt_class, "KRBLEVEL" ).setInteger(CURLOPT_KRBLEVEL).setReadOnly(true);
   self->addClassProperty( copt_class, "SSH_HOST_PUBLIC_KEY_MD5" ).setInteger(CURLOPT_SSH_HOST_PUBLIC_KEY_MD5).setReadOnly(true);
   self->addClassProperty( copt_class, "SSH_PUBLIC_KEYFILE" ).setInteger(CURLOPT_SSH_PUBLIC_KEYFILE).setReadOnly(true);
   self->addClassProperty( copt_class, "SSH_PRIVATE_KEYFILE" ).setInteger(CURLOPT_SSH_PRIVATE_KEYFILE).setReadOnly(true);

#ifdef CURLOPT_SSH_KNOWNHOSTS
   self->addClassProperty( copt_class, "SSH_KNOWNHOSTS" ).setInteger(CURLOPT_SSH_KNOWNHOSTS).setReadOnly(true);
#endif

   // List options
   self->addClassProperty( copt_class, "HTTPHEADER" ).setInteger(CURLOPT_HTTPHEADER).setReadOnly(true);
   self->addClassProperty( copt_class, "HTTP200ALIASES" ).setInteger(CURLOPT_HTTP200ALIASES).setReadOnly(true);
   self->addClassProperty( copt_class, "QUOTE" ).setInteger(CURLOPT_QUOTE).setReadOnly(true);
   self->addClassProperty( copt_class, "POSTQUOTE" ).setInteger(CURLOPT_POSTQUOTE).setReadOnly(true);
   self->addClassProperty( copt_class, "PREQUOTE" ).setInteger(CURLOPT_PREQUOTE).setReadOnly(true);

   // To be implemented separately
   /*
   CURLOPT_HTTPPOST

   CURLOPT_SSH_KEYFUNCTION
   CURLOPT_SSH_KEYDATA

   CURLOPT_SHARE (?)

   CURLOPT_TELNETOPTIONS
   CURLOPT_TIMECONDITION
   */

   //============================================================
   // Enumeration class CURLINFO
   //
   Falcon::Symbol *curlinfo_class = self->addClass( "INFO" );
   self->addClassProperty( curlinfo_class, "EFFECTIVE_URL" ).setInteger(CURLINFO_EFFECTIVE_URL).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "RESPONSE_CODE" ).setInteger(CURLINFO_RESPONSE_CODE).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "HTTP_CONNECTCODE" ).setInteger(CURLINFO_HTTP_CONNECTCODE).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "FILETIME" ).setInteger(CURLINFO_FILETIME).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "TOTAL_TIME" ).setInteger(CURLINFO_TOTAL_TIME).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "NAMELOOKUP_TIME" ).setInteger(CURLINFO_NAMELOOKUP_TIME).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "CONNECT_TIME" ).setInteger(CURLINFO_CONNECT_TIME).setReadOnly(true);
#if LIBCURL_VERSION_NUM >= 0x071900
   self->addClassProperty( curlinfo_class, "APPCONNECT_TIME" ).setInteger(CURLINFO_APPCONNECT_TIME).setReadOnly(true);
#endif
   self->addClassProperty( curlinfo_class, "PRETRANSFER_TIME" ).setInteger(CURLINFO_PRETRANSFER_TIME).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "STARTTRANSFER_TIME" ).setInteger(CURLINFO_STARTTRANSFER_TIME).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "REDIRECT_TIME" ).setInteger(CURLINFO_REDIRECT_TIME).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "REDIRECT_COUNT" ).setInteger(CURLINFO_REDIRECT_COUNT).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "REDIRECT_URL" ).setInteger(CURLINFO_REDIRECT_URL).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "SIZE_UPLOAD" ).setInteger(CURLINFO_SIZE_UPLOAD).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "SIZE_DOWNLOAD" ).setInteger(CURLINFO_SIZE_DOWNLOAD).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "SPEED_DOWNLOAD" ).setInteger(CURLINFO_SPEED_DOWNLOAD).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "SPEED_UPLOAD" ).setInteger(CURLINFO_SPEED_UPLOAD).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "HEADER_SIZE" ).setInteger(CURLINFO_HEADER_SIZE).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "REQUEST_SIZE" ).setInteger(CURLINFO_REQUEST_SIZE).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "SSL_VERIFYRESULT" ).setInteger(CURLINFO_SSL_VERIFYRESULT).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "SSL_ENGINES" ).setInteger(CURLINFO_SSL_ENGINES).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "CONTENT_LENGTH_DOWNLOAD" ).setInteger(CURLINFO_CONTENT_LENGTH_DOWNLOAD).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "CONTENT_LENGTH_UPLOAD" ).setInteger(CURLINFO_CONTENT_LENGTH_UPLOAD).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "CONTENT_TYPE" ).setInteger(CURLINFO_CONTENT_TYPE).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "HTTPAUTH_AVAIL" ).setInteger(CURLINFO_HTTPAUTH_AVAIL).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "PROXYAUTH_AVAIL" ).setInteger(CURLINFO_PROXYAUTH_AVAIL).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "NUM_CONNECTS" ).setInteger(CURLINFO_NUM_CONNECTS).setReadOnly(true);
#if LIBCURL_VERSION_NUM >= 0x071900
   self->addClassProperty( curlinfo_class, "PRIMARY_IP" ).setInteger(CURLINFO_PRIMARY_IP).setReadOnly(true);
#endif
   self->addClassProperty( curlinfo_class, "COOKIELIST" ).setInteger(CURLINFO_COOKIELIST).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "FTP_ENTRY_PATH" ).setInteger(CURLINFO_FTP_ENTRY_PATH).setReadOnly(true);
   self->addClassProperty( curlinfo_class, "SSL_ENGINES" ).setInteger(CURLINFO_SSL_ENGINES).setReadOnly(true);
#if LIBCURL_VERSION_NUM >= 0x071904
   self->addClassProperty( curlinfo_class, "CONDITION_UNMET" ).setInteger(CURLINFO_CONDITION_UNMET).setReadOnly(true);
#endif

   /**
    * Separately handled
    *    CURLINFO_PRIVATE -> CURLOPT_PRIVATE
    *    CURLINFO_LASTSOCKET -> socket?
    *    CURLINFO_CERTINFO
    */

   //============================================================
   // CurlError class
   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *ce_cls = self->addClass( "CurlError", Falcon::Ext::CurlError_init );
   ce_cls->setWKS( true );
   ce_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );


   return self;
}

/* end of curl.cpp */
