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

#define FALCON_ENGINE_STATIC

#include <falcon/setup.h>
#include <falcon/error.h>
#include <falcon/class.h>
#include <falcon/function.h>

#ifndef FALCON_REGEX_ERROR_BASE
   #define FALCON_REGEX_ERROR_BASE        1160
#endif

#define FALRE_ERR_INVALID    (FALCON_REGEX_ERROR_BASE + 0)
#define FALRE_ERR_STUDY      (FALCON_REGEX_ERROR_BASE + 1)
#define FALRE_ERR_ERRMATCH   (FALCON_REGEX_ERROR_BASE + 2)

namespace Falcon {
namespace Ext {

class ClassRegex : public Class
{
public:
   ClassRegex();
   virtual ~ClassRegex();

   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
   virtual void* createInstance() const; 

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;


private:

   FALCON_DECLARE_FUNCTION(study, "");
   FALCON_DECLARE_FUNCTION(match, "");
   FALCON_DECLARE_FUNCTION(grab, "");
   FALCON_DECLARE_FUNCTION(split, "S, [N], [B]");
   FALCON_DECLARE_FUNCTION(find, "");
   FALCON_DECLARE_FUNCTION(findAll, "S, [I], [I]");
   FALCON_DECLARE_FUNCTION(findAllOverlapped, "S, [I], [I]");
   FALCON_DECLARE_FUNCTION(replace, "");
   FALCON_DECLARE_FUNCTION(replaceAll, "");
   FALCON_DECLARE_FUNCTION(subst, "");
   FALCON_DECLARE_FUNCTION(capturedCount, "");
   FALCON_DECLARE_FUNCTION(captured, "");
   FALCON_DECLARE_FUNCTION(compare, "");
   FALCON_DECLARE_FUNCTION(version, "");

};

FALCON_DECLARE_ERROR( RegexError )

}
}

#endif

/* end of regex_ext.h */
