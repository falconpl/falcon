#ifndef G_OBJECT_HPP
#define G_OBJECT_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Glib {

/**
 *  \namespace Falcon::Glib::Object
 */
namespace Object {

void modInit( Falcon::Module* );

FALCON_FUNC signal_notify( VMARG );

void on_notify( GObject*, GParamSpec*, gpointer );


} // Object
} // Glib
} // Falcon

#endif // !G_OBJECT_HPP
