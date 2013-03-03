/*
   FALCON - The Falcon Programming Language.
   FILE: regex_ext.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab mar 11 2006

   Process module -- Falcon interface functions
   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Regular expression module -- Falcon interface functions
   This is the module declaration file.
*/

#ifndef flc_regex_ext_H
#define flc_regex_ext_H

#include <falcon/setup.h>

#include <falcon/error.h>

#include <falcon/classes/classerror.h>
#include <falcon/classes/classuser.h>
#include <falcon/method.h>

#ifndef FALCON_REGEX_ERROR_BASE
   #define FALCON_REGEX_ERROR_BASE        1160
#endif

#define FALRE_ERR_INVALID    (FALCON_REGEX_ERROR_BASE + 0)
#define FALRE_ERR_STUDY      (FALCON_REGEX_ERROR_BASE + 1)
#define FALRE_ERR_ERRMATCH   (FALCON_REGEX_ERROR_BASE + 2)

namespace Falcon {
namespace Ext {

class ClassRegex : public ClassUser
{
public:
   ClassRegex();
   virtual ~ClassRegex();

   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
   virtual void* createInstance() const; 

private:

FALCON_DECLARE_METHOD(study, "");
FALCON_DECLARE_METHOD(match, "");
FALCON_DECLARE_METHOD(grab, "");
FALCON_DECLARE_METHOD(split, "S, [N], [B]");
FALCON_DECLARE_METHOD(find, "");
FALCON_DECLARE_METHOD(findAll, "S, [I], [I]");
FALCON_DECLARE_METHOD(findAllOverlapped, "S, [I], [I]");
FALCON_DECLARE_METHOD(replace, "");
FALCON_DECLARE_METHOD(replaceAll, "");
FALCON_DECLARE_METHOD(subst, "");
FALCON_DECLARE_METHOD(capturedCount, "");
FALCON_DECLARE_METHOD(captured, "");
FALCON_DECLARE_METHOD(compare, "");
FALCON_DECLARE_METHOD(version, "");

};

class ClassRegexError: public ClassError
      {
      private:
         static ClassRegexError* m_instance;
      public:
         inline ClassRegexError(): ClassError( "RegexError" ) {} 
         inline virtual ~ClassRegexError(){} 
         virtual void* createInstance() const;
         static ClassRegexError* singleton();
      };

class RegexError: public ::Falcon::Error
{
public:
   RegexError():
      Error( ClassRegexError::singleton() )
   {}

   RegexError( const ErrorParam &params  ):
      Error( ClassRegexError::singleton(), params )
      {}
};

}
}

#endif

/* end of regex_ext.h */
