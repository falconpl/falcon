#ifndef GTK_HPANED_HPP
#define GTK_HPANED_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::HPaned
 */
class HPaned
    :
    public Gtk::CoreGObject
{
public:

    HPaned( const Falcon::CoreClass*, const GtkHPaned* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_HPANED_HPP
