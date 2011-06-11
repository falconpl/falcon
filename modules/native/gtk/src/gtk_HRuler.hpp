#ifndef GTK_HRULER_HPP
#define GTK_HRULER_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::HRuler
 */
class HRuler
    :
    public Gtk::CoreGObject
{
public:

    HRuler( const Falcon::CoreClass*, const GtkHRuler* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_HRULER_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
