#ifndef GTK_PROGRESS_HPP
#define GTK_PROGRESS_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Progress
 */
class Progress
    :
    public Gtk::CoreGObject
{
public:

    Progress( const Falcon::CoreClass*, const GtkProgress* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );
#if 0 // Deprecated
    static FALCON_FUNC set_show_text( VMARG );

    static FALCON_FUNC set_text_alignment( VMARG );

    static FALCON_FUNC set_format_string( VMARG );

    static FALCON_FUNC set_adjustment( VMARG );

    static FALCON_FUNC set_percentage( VMARG );

    static FALCON_FUNC set_value( VMARG );

    static FALCON_FUNC get_value( VMARG );

    static FALCON_FUNC set_activity_mode( VMARG );

    static FALCON_FUNC get_current_text( VMARG );

    static FALCON_FUNC get_text_from_value( VMARG );

    static FALCON_FUNC get_current_percentage( VMARG );

    static FALCON_FUNC get_percentage_from_value( VMARG );

    static FALCON_FUNC configure( VMARG );
#endif
};


} // Gtk
} // Falcon

#endif // !GTK_PROGRESS_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
