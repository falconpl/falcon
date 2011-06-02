#ifndef GTK_TEXTMARK_HPP
#define GTK_TEXTMARK_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TextMark
 */
class TextMark
    :
    public Gtk::CoreGObject
{
public:

    TextMark( const Falcon::CoreClass*, const GtkTextMark* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC set_visible( VMARG );

    static FALCON_FUNC get_visible( VMARG );

    static FALCON_FUNC get_deleted( VMARG );

    static FALCON_FUNC get_name( VMARG );

    static FALCON_FUNC get_buffer( VMARG );

    static FALCON_FUNC get_left_gravity( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TEXTMARK_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
