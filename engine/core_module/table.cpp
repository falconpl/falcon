/*
   FALCON - The Falcon Programming Language.
   FILE: table.cpp

   Table support Iterface for Falcon.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 14 Sep 2008 15:55:59 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


namespace Falcon {

FALCON_FUNC Table_init( VMachine* vm )
{
   Item *i_tarr = vm->param( 0 );
   if ( i_var == 0 || ! i_var->isArray() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "A" ) ) );
      return;
   }

   // More; also the first element of the array must be an array.
   CoreArray *tarr = i_tarr->asArray();

   Item *i_heading;
   if ( tarr->length() != 0
        || ! (i_heading = &tarr->at( 0 ))->isArray()
        || i_heading->asArray()->length() == 0
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( FAL_STR(rtl_no_tabhead) ) ) );
      return;
   }


}

}

/* end of table.cpp */
