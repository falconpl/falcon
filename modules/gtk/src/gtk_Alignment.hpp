#ifndef GTK_ALIGNMENT_HPP
#define GTK_ALIGNMENT_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Alignment
 */
class Alignment
    :
    public Gtk::CoreGObject
{
public:

    Alignment( const Falcon::CoreClass*, const GtkAlignment* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC set( VMARG );

    static FALCON_FUNC get_padding( VMARG );

    static FALCON_FUNC set_padding( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_ALIGNMENT_HPP
