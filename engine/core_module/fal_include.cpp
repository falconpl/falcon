/*
   FALCON - The Falcon Programming Language.
   FILE: fal_include.cpp

   Include function - Dynamic module loading.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 31 Aug 2008 19:01:59 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/vm.h>
#include <falcon/module.h>

#include <falcon/flcloader.h>

namespace Falcon {
namespace core {

/*#
   @function include
   @brief Dynamically loads a module as a plugin.
   @param file The relative filename of the module.
   @optparam inputEnc Input encoding.
   @optparam path A string of ';' separated search paths.
   @optparam symDict Symbols to be queried (or nil).

   @raise IOError if the module cannot be found or load.

   A module indicated by filename is compiled, loaded and linked in the
   running Virtual Machine. The inclusion is relative to the current
   path, be it set in the script, in the current embedding application
   or in the falcon command line interpreter. It is possible to use
   a path relative to the current script path by using the scriptPath
   variable.

   If a dictionary of symbol to be queried is @b not provided, the module
   is loaded and its main code, if present, is executed.

   If @b symDict is provided, its keys are strings which refer to symbol names
   to be searched in the loaded module. If present, the entry are filled with
   symbols coming from the loaded module. When @b symDict is provided, the
   linked module won't have its main code executed (it is possible to execute
   it at a later time adding "__main__" to the retreived modules). If a symbol
   is not found, its entry in the dictionary will be set to nil. When loaded
   this way, the export requests in the loaded module are @b not honored (import/from
   semantic).

   The @b compiler Feather module provides a more complete interface to dynamic
   load and compilation, but this minimal dynamic load support is provided at
   base level for flexibility.
*/

FALCON_FUNC fal_include( Falcon::VMachine *vm )
{
   Falcon::Item *i_file = vm->param(0);
   Falcon::Item *i_enc = vm->param(1);
   Falcon::Item *i_path = vm->param(2);
   Falcon::Item *i_syms = vm->param(3);

   if( i_file == 0 || ! i_file->isString()
      || (i_syms != 0 && ! (i_syms->isDict() || i_syms->isNil())  )
      || (i_enc != 0 && !(i_enc->isString() || i_syms->isNil()) )
      || (i_path != 0 && !(i_path->isString() || i_syms->isNil()) )
      )
   {
      vm->raiseModError( new Falcon::ParamError(
         Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ ).
         extra( "S,[S],[S],[D]" ) ) );
      return;
   }

   // create the loader/runtime pair.
   FlcLoader cpl( i_path == 0 || i_path->isNil() ? "." : *i_path->asString() );
   Runtime rt( &cpl, vm );
   rt.hasMainModule( false );

   // minimal config
   cpl.errorHandler( vm->errorHandler() );
   if ( i_enc != 0 && ! i_enc->isNil() )
   {
      cpl.sourceEncoding( *i_enc->asString() );
   }

   // load and link
   bool execAtLink = vm->launchAtLink();
   if ( rt.loadFile( *i_file->asString(), false ) )
   {
      vm->launchAtLink( i_syms == 0 || i_syms->isNil() );
      LiveModule *lmod = vm->link( &rt );

      // shall we read the symbols?
      if( ! vm->hadError() && lmod != 0 && ( i_syms != 0 && i_syms->isDict() ) )
      {
         CoreDict *dict = i_syms->asDict();

         // get the topmost module.
         uint32 listSize = rt.moduleVector()->size();
         ModuleDep *md = rt.moduleVector()->moduleDepAt( listSize-1 );
         vm->findModule( md->module()->name() );
         fassert( lmod != 0 );

         // traverse the dictionary
         DictIterator *iter = dict->first();
         while( iter->isValid() )
         {
            // if the key is a string and a coresponding item is found...
            Item *ival;
            if ( iter->getCurrentKey().isString() &&
                 ( ival = lmod->findModuleItem( *iter->getCurrentKey().asString() ) ) != 0 )
            {
               // copy it locally
               iter->getCurrent() = *ival;
            }
            else {
               iter->getCurrent().setNil();
            }

            iter->next();
         }
         delete iter;
      }
   }

   // reset launch status
   vm->launchAtLink( execAtLink );
}

}
}
