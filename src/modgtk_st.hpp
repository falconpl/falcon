/**
 *  \file modgtk_st.hpp
 */

//WARNING: the missing of usual #ifndef/#define pair
//         is intentional!

// See the contents of this header file for a deeper overall
// explanation of the MODSTR system.
#include <falcon/message_defs.h>

// The first parameter is an unique identifier in your project that
// will be bound to the correct entry in the module string table.
// Falcon::VMachine::moduleString( @PROJECT_NAME@_msg_1 ) will
// return the associated string or the internationalized version.
// FAL_STR( @PROJECT_NAME@_msg_1 ) macro can be used in standard
// functions as a shortcut.

FAL_MODSTR( gtk_msg_1, "Falcon bindings to GTK+" );

//... add here your messages, and remove or configure the above one

/*
 *  messages for gtk errors (one by error id)
 *  add a trailing underscore to avoid names collision
 */
FAL_MODSTR( gtk_e_abstract_class_,      "unable to create instance of abstract class" );
FAL_MODSTR( gtk_e_init_failure_,        "init function failure" );

/*
 *  messages for param errors
 */
FAL_MODSTR( gtk_e_require_no_args_,     "this method does not accept arguments" );
FAL_MODSTR( gtk_e_inv_window_type_,     "invalid window type" );
FAL_MODSTR( gtk_e_inv_property_,        "invalid object property" );
