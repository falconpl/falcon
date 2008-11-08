/*
   The Falcon Programming Language
   FILE: dynlib_st.h

   Direct dynamic library interface for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Oct 2008 22:23:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: The Falcon Comittee

   See the LICENSE file distributed with this package for licensing details.
*/

/** \file
   String table - export internationalizable stings to both C++
                  and Falcon modules.
*/

//WARNING: the missing of usual #ifndef/#define pair
//         is intentional!

// See the contents of this header file for a deeper overall
// explanation of the MODSTR system.
#include <falcon/message_defs.h>

// The first parameter is an unique identifier in your project that
// will be bound to the correct entry in the module string table.
// Falcon::VMachine::moduleString( dynlib_msg_1 ) will
// return the associated string or the internationalized version.
// FAL_STR( dynlib_msg_1 ) macro can be used in standard
// functions as a shortcut.

FAL_MODSTR( dle_load_error, "Error in loading the dynamic library" );
FAL_MODSTR( dle_unknown_error, "Reason unknown" );
FAL_MODSTR( dle_already_unloaded, "Library has been already unloaded" );
FAL_MODSTR( dle_unload_fail, "Failed to unload the library" );
FAL_MODSTR( dle_cant_instance, "Can't create directly an instance of class DynFunction" );
FAL_MODSTR( dle_cant_describe, "Error description unavailable" );
FAL_MODSTR( dle_cant_guess_param, "Cannot guess automatic conversion for parameter" );
FAL_MODSTR( dle_symbol_not_found, "Symbol not found" );
FAL_MODSTR( dyl_toomany_pars, "Too many parameters" );
FAL_MODSTR( dyl_invalid_pmask, "Invalid parameter mask" );
FAL_MODSTR( dyl_invalid_rmask, "Invalid return mask" );
FAL_MODSTR( dyl_param_mismatch, "Given parameter type is not matching formal definition." );
FAL_MODSTR( dle_not_byref, "Parameter is declared as reference but passed by value." );

/* end of dynlib_st.h */
