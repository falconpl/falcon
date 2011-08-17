/*
   FALCON - The Falcon Programming Language.
   FILE: message_defs.h

   Definitions for messages.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin:

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


/**
   @page howto_modstrings How to use module string tables.

   The module string table is useful to declare error descriptions,
   error specific explanations and generically messages that the module
   may display to its users.

   Module message tables can be internationalized through the Falcon
   Module Internationalization support.

   Some macros are provided in module.h to help build module string tables,
   and are meant to be used under a certain pattern.

   Applying the module table to the module is a matter of four simple steps.

   First, each module willing to create an internationalizable module table should
   create two related files: \<modulename\>_st.h and \<modulename\>_st.c(pp)

   Second, create the table declaring it in the header file using the FAL_MODSTR
   macro, like in the following example:
   \code
      // My module string table mymod_st.h
      // Message IDS (identifiers) must be unique.

      #include \<falcon/message_defs.h\>

      FAL_MODSTR( MSG0, "Message 0" );
      FAL_MODSTR( MSG1, "Message 1" );
      FAL_MODSTR( MSG2, "Message 2" );
   \endcode

   If the program is being compiled  in c++ mode, it is possible to declare a
   namespace around the FAL_MODSTR marcors for better encapsulation. The semicomma ";"
   at the end of each macro are optional.

   Second, write the C/C++ table implementation. This is only required to
   declare the macro FALCON_REALIZE_STRTAB before including the string table definition:
   \code
      // Module string table realize file mymod_st.cpp

      #define FALCON_REALIZE_STRTAB
      #include "mymod_st.h"
   \endcode

   Third, the main module file (usually called something as \<modname\>.cpp) must
   first include the string table at top level, and then realize it
   by declaring FALCON_DECLARE_MODULE, setting it
   to the local variable pointer used to instance the module, and then
   include the string table:

   \code
   #include \<module.h\>
   #include "mymod_st.h"

   FALCON_MODULE_DECL( const Falcon::EngineData &data )
   {
      // setup DLL engine common data
      data.set();

      // Module declaration
      Falcon::Module *self = new Falcon::Module();

      // Declare "self" as the variable holding the module
      #define FALCON_DECLARE_MODULE self
      #include "mymod_st.h"
      ...
   }
   \endcode

   Fourth, retreive the strings from the VM using the Falcon::VMachine::moduleString
   method. That method actually peeks the current module for the desired string id.
   In example:

   \code
   #include "mymod_st.h"

   FALCON_FUNC aFuncIn_MyModule( Falcon::VMachine *vm )
   {
      const String *tlstring = vm->moduleString( MSG0 );
      // do something with tlstring
   }
   \endcode

   The same can be achieved with Module::getString provided it is possible to access
   the module:
   \code
   #include "mymod_st.h"

   // MyNewModule extends Falcon::Module
   void MyNewModule::some_method(...)
   {
      const String *tlstring = this->getString( MSG0 );
      // do something with tlstring
   }
   \endcode

   The macro FAL_STR( id ), defined as "vm->moduleString( id )" can be used as a handy
   shortcut for a standard compliant extension function.
*/
#ifndef FAL_STR
   #define FAL_STR( id )   vm->moduleString( id )
#endif

// allow redefinition of this macros at each include.
#undef FAL_MODSTR
#ifdef FALCON_DECLARE_MODULE
   #define FAL_MODSTR( str_id, text ) \
      str_id = FALCON_DECLARE_MODULE->addStringID( text, true );
#else
   #ifdef FALCON_REALIZE_STRTAB
      #define FAL_MODSTR( id, text )   unsigned int id;
   #else
      #define FAL_MODSTR( id, text )   extern unsigned int id;
   #endif
#endif

/* end of message_defs.h */
