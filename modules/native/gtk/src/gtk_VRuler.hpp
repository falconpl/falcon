#ifndef GTK_VRULER_HPP
#define GTK_VRULER_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::VRuler
 */
class VRuler
    :
    public Gtk::CoreGObject
{
public:

    VRuler( const Falcon::CoreClass*, const GtkVRuler* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_VRULER_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
