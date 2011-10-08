/**
 *  \file gtk_AccelMap.hpp
 */

#ifndef GTK_ACCELMAP_HPP
#define GTK_ACCELMAP_HPP

#include "modgtk.hpp"

namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::AccelMap
 */
class AccelMap
    :
    public Gtk::CoreGObject
{
public:

    AccelMap( const Falcon::CoreClass*, const GtkAccelMap* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC add_entry( VMARG );

};

} // Gtk
} // Falcon

#endif // !GTK_ACCELMAP_HPP

// vi: set ai et sw=4 ts=4 sts=4:
// kate: replace-tabs on; shift-width 4;
