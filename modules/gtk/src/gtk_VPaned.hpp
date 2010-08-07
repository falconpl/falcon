#ifndef GTK_VPANED_HPP
#define GTK_VPANED_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::VPaned
 */
class VPaned
    :
    public Gtk::CoreGObject
{
public:

    VPaned( const Falcon::CoreClass*, const GtkVPaned* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_VPANED_HPP
