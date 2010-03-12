/**
 *  \file gtk_Label.cpp
 */

#include "gtk_Label.hpp"

#include "gtk_Widget.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Label::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Label = mod->addClass( "Label", &Label::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "Widget" ) );
    c_Label->getClassDef()->addInheritance( in );

    Gtk::MethodTab methods[] =
    {
    { "set_text",               &Label::set_text },
//    { "set_attributes",       &Label::set_attributes },
    { "set_markup",             &Label::set_markup },
    { "set_markup_with_mnemonic",&Label::set_markup_with_mnemonic },
    { "set_pattern",            &Label::set_pattern },
//    { "set_justify",          &Label::set_justify },
//    { "set_ellipsize",        &Label::set_ellipsize },
    { "set_width_chars",        &Label::set_width_chars },
    { "set_max_width_chars",    &Label::set_max_width_chars },
//    { "get",                  &Label::get },
//    { "parse_uline",          &Label::parse_uline },
    { "set_line_wrap",          &Label::set_line_wrap },
//    { "set_line_wrap_mode",   &Label::set_line_wrap_mode },
//    { "set",                  &Label::set },
//    { "get_layout_offsets",   &Label::get_layout_offsets },
    { "get_mnemonic_keyval",    &Label::get_mnemonic_keyval },
    { "get_selectable",         &Label::get_selectable },
    { "get_text",               &Label::get_text },
//    { "new_with_mnemonic",    &Label::new_with_mnemonic },
    { "select_region",          &Label::select_region },
    { "set_mnemonic_widget",    &Label::set_mnemonic_widget },
    { "set_selectable",         &Label::set_selectable },
    { "set_text_with_mnemonic", &Label::set_text_with_mnemonic },
//    { "get_attributes",       &Label::get_attributes },
//    { "get_justify",          &Label::get_justify },
//    { "get_ellipsize",        &Label::get_ellipsize },
    { "get_width_chars",        &Label::get_width_chars },
    { "get_max_width_chars",    &Label::get_max_width_chars },
    { "get_label",              &Label::get_label },
//    { "get_layout",           &Label::get_layout },
    { "get_line_wrap",          &Label::get_line_wrap },
//    { "get_line_wrap_mode",   &Label::get_line_wrap_mode },
    { "get_mnemonic_widget",    &Label::get_mnemonic_widget },
//    { "get_selection_bounds", &Label::get_selection_bounds },
    { "get_use_markup",         &Label::get_use_markup },
    { "get_use_underline",      &Label::get_use_underline },
    { "get_single_line_mode",   &Label::get_single_line_mode },
    { "get_angle",              &Label::get_angle },
    { "set_label",              &Label::set_label },
    { "set_use_markup",         &Label::set_use_markup },
    { "set_use_underline",      &Label::set_use_underline },
    { "set_single_line_mode",   &Label::set_single_line_mode },
    { "set_angle",              &Label::set_angle },
#if GTK_VERSION_MINOR >= 18
    { "get_current_uri",        &Label::get_current_uri },
    { "set_track_visited_links",&Label::set_track_visited_links },
    { "get_track_visited_links",&Label::get_track_visited_links },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Label, meth->name, meth->cb );
}

/*#
    @class gtk.Label
    @brief The GtkLabel widget displays a small amount of text.

    As the name implies, most labels are used to label another widget such as
    a GtkButton, a GtkMenuItem, or a GtkOptionMenu.
 */

/*#
    @init gtk.Label
    @brief Create a label (optionaly with mnemonic)
    @optparam label (string)
    @optparam mnemonic (boolean, default false)
 */
FALCON_FUNC Label::init( VMARG )
{
    MYSELF;

    if ( self->getUserData() )
        return;

    GtkWidget* gwdt;

    Item* i_lbl = vm->param( 0 );
    if ( i_lbl )
    {
        if ( i_lbl->isNil() || !i_lbl->isString() )
            throw_inv_params( "[S[,B]]" );

        AutoCString s( i_lbl->asString() );
        bool mnemo = false;

        Item* i_mnemo = vm->param( 1 );
        if ( i_mnemo )
        {
            if ( i_mnemo->isNil() || !i_mnemo->isBoolean() )
                throw_inv_params( "[S[,B]]" );
            mnemo = i_mnemo->asBoolean();
        }

        gwdt = mnemo ? gtk_label_new_with_mnemonic( s.c_str() )
            : gtk_label_new( s.c_str() );
    }
    else
        gwdt = gtk_label_new( NULL );

    Gtk::internal_add_slot( (GObject*) gwdt );
    self->setUserData( new GData( (GObject*) gwdt ) );
}


FALCON_FUNC Label::set_text( VMARG )
{
    Item* i_txt = vm->param( 0 );
    if ( !i_txt || i_txt->isNil() || !i_txt->isString() )
        throw_inv_params( "S" );
    AutoCString s( i_txt->asString() );
    MYSELF;
    GET_OBJ( self );
    gtk_label_set_text( (GtkLabel*)_obj, s.c_str() );
}


//FALCON_FUNC Label::set_attributes( VMARG );


FALCON_FUNC Label::set_markup( VMARG )
{
    Item* i_txt = vm->param( 0 );
    if ( !i_txt || i_txt->isNil() || !i_txt->isString() )
        throw_inv_params( "S" );
    AutoCString s( i_txt->asString() );
    MYSELF;
    GET_OBJ( self );
    gtk_label_set_markup( (GtkLabel*)_obj, s.c_str() );
}


FALCON_FUNC Label::set_markup_with_mnemonic( VMARG )
{
    Item* i_txt = vm->param( 0 );
    if ( !i_txt || i_txt->isNil() || !i_txt->isString() )
        throw_inv_params( "S" );
    AutoCString s( i_txt->asString() );
    MYSELF;
    GET_OBJ( self );
    gtk_label_set_markup_with_mnemonic( (GtkLabel*)_obj, s.c_str() );
}


FALCON_FUNC Label::set_pattern( VMARG )
{
    Item* i_txt = vm->param( 0 );
    if ( !i_txt || i_txt->isNil() || !i_txt->isString() )
        throw_inv_params( "S" );
    AutoCString s( i_txt->asString() );
    MYSELF;
    GET_OBJ( self );
    gtk_label_set_pattern( (GtkLabel*)_obj, s.c_str() );
}


//FALCON_FUNC Label::set_justify( VMARG );

//FALCON_FUNC Label::set_ellipsize( VMARG );


FALCON_FUNC Label::set_width_chars( VMARG )
{
    Item* i_nchars = vm->param( 0 );
    if ( !i_nchars || i_nchars->isNil() || !i_nchars->isInteger() )
        throw_inv_params( "I" );
    MYSELF;
    GET_OBJ( self );
    gtk_label_set_width_chars( (GtkLabel*)_obj, i_nchars->asInteger() );
}


FALCON_FUNC Label::set_max_width_chars( VMARG )
{
    Item* i_nchars = vm->param( 0 );
    if ( !i_nchars || i_nchars->isNil() || !i_nchars->isInteger() )
        throw_inv_params( "I" );
    MYSELF;
    GET_OBJ( self );
    gtk_label_set_max_width_chars( (GtkLabel*)_obj, i_nchars->asInteger() );
}


//FALCON_FUNC Label::get( VMARG ) deprecated

//FALCON_FUNC Label::parse_uline( VMARG ) deprecated


FALCON_FUNC Label::set_line_wrap( VMARG )
{
    Item* i_bool = vm->param( 0 );
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
    MYSELF;
    GET_OBJ( self );
    gtk_label_set_line_wrap( (GtkLabel*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


//FALCON_FUNC Label::set_line_wrap_mode( VMARG );

//FALCON_FUNC Label::set( VMARG ); deprecated

//FALCON_FUNC Label::get_layout_offsets( VMARG );


FALCON_FUNC Label::get_mnemonic_keyval( VMARG )
{
    if ( vm->paramCount() )
        throw_require_no_args();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_label_get_mnemonic_keyval( (GtkLabel*)_obj ) );
}


FALCON_FUNC Label::get_selectable( VMARG )
{
    if ( vm->paramCount() )
        throw_require_no_args();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_label_get_selectable( (GtkLabel*)_obj ) );
}


FALCON_FUNC Label::get_text( VMARG )
{
    if ( vm->paramCount() )
        throw_require_no_args();
    MYSELF;
    GET_OBJ( self );
    vm->retval( new String( gtk_label_get_text( (GtkLabel*)_obj ) ) );
}


//FALCON_FUNC Label::new_with_mnemonic( VMARG );


FALCON_FUNC Label::select_region( VMARG )
{
    Item* i_startOffset = vm->param( 0 );
    Item* i_endOffset = vm->param( 1 );
    if ( !i_startOffset || i_startOffset->isNil() || !i_startOffset->isInteger()
        || !i_endOffset || i_endOffset->isNil() || !i_endOffset->isInteger() )
        throw_inv_params( "I,I" );
    MYSELF;
    GET_OBJ( self );
    gtk_label_select_region( (GtkLabel*)_obj,
        i_startOffset->asInteger(), i_endOffset->asInteger() );
}


FALCON_FUNC Label::set_mnemonic_widget( VMARG )
{
    Item* i_wdt = vm->param( 0 );
    if ( !i_wdt || i_wdt->isNil() ||
        !( i_wdt->isOfClass( "Widget" ) || i_wdt->isOfClass( "gtk.Widget" ) ) )
        throw_inv_params( "Widget" );
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = (GtkWidget*)((GData*)i_wdt->asObject()->getUserData())->obj();
    gtk_label_set_mnemonic_widget( (GtkLabel*)_obj, wdt );
}


FALCON_FUNC Label::set_selectable( VMARG )
{
    Item* i_bool = vm->param( 0 );
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
    MYSELF;
    GET_OBJ( self );
    gtk_label_set_selectable( (GtkLabel*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


FALCON_FUNC Label::set_text_with_mnemonic( VMARG )
{
    Item* i_txt = vm->param( 0 );
    if ( !i_txt || i_txt->isNil() || !i_txt->isString() )
        throw_inv_params( "S" );
    AutoCString s( i_txt->asString() );
    MYSELF;
    GET_OBJ( self );
    gtk_label_set_text_with_mnemonic( (GtkLabel*)_obj, s.c_str() );
}


//FALCON_FUNC Label::get_attributes( VMARG );

//FALCON_FUNC Label::get_justify( VMARG );

//FALCON_FUNC Label::get_ellipsize( VMARG );


FALCON_FUNC Label::get_width_chars( VMARG )
{
    if ( vm->paramCount() )
        throw_require_no_args();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_label_get_width_chars( (GtkLabel*)_obj ) );
}


FALCON_FUNC Label::get_max_width_chars( VMARG )
{
    if ( vm->paramCount() )
        throw_require_no_args();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_label_get_max_width_chars( (GtkLabel*)_obj ) );
}


FALCON_FUNC Label::get_label( VMARG )
{
    if ( vm->paramCount() )
        throw_require_no_args();
    MYSELF;
    GET_OBJ( self );
    vm->retval( new String( gtk_label_get_label( (GtkLabel*)_obj ) ) );
}


//FALCON_FUNC Label::get_layout( VMARG );


FALCON_FUNC Label::get_line_wrap( VMARG )
{
    if ( vm->paramCount() )
        throw_require_no_args();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_label_get_line_wrap( (GtkLabel*)_obj ) );
}


//FALCON_FUNC Label::get_line_wrap_mode( VMARG );


FALCON_FUNC Label::get_mnemonic_widget( VMARG )
{
    if ( vm->paramCount() )
        throw_require_no_args();
    MYSELF;
    GET_OBJ( self );
    GtkWidget* gwdt = gtk_label_get_mnemonic_widget( (GtkLabel*)_obj );
    if ( gwdt )
    {
        Item* wki = vm->findWKI( "Widget" );
        vm->retval( new Gtk::Widget( wki->asClass(), gwdt ) );
    }
    else
        vm->retnil();
}


//FALCON_FUNC Label::get_selection_bounds( VMARG );


FALCON_FUNC Label::get_use_markup( VMARG )
{
    if ( vm->paramCount() )
        throw_require_no_args();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_label_get_use_markup( (GtkLabel*)_obj ) );
}


FALCON_FUNC Label::get_use_underline( VMARG )
{
    if ( vm->paramCount() )
        throw_require_no_args();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_label_get_use_underline( (GtkLabel*)_obj ) );
}


FALCON_FUNC Label::get_single_line_mode( VMARG )
{
    if ( vm->paramCount() )
        throw_require_no_args();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_label_get_single_line_mode( (GtkLabel*)_obj ) );
}


FALCON_FUNC Label::get_angle( VMARG )
{
    if ( vm->paramCount() )
        throw_require_no_args();
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_label_get_angle( (GtkLabel*)_obj ) );
}


FALCON_FUNC Label::set_label( VMARG )
{
    Item* i_txt = vm->param( 0 );
    if ( !i_txt || i_txt->isNil() || !i_txt->isString() )
        throw_inv_params( "S" );
    AutoCString s( i_txt->asString() );
    MYSELF;
    GET_OBJ( self );
    gtk_label_set_label( (GtkLabel*)_obj, s.c_str() );
}


FALCON_FUNC Label::set_use_markup( VMARG )
{
    Item* i_bool = vm->param( 0 );
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
    MYSELF;
    GET_OBJ( self );
    gtk_label_set_use_markup( (GtkLabel*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


FALCON_FUNC Label::set_use_underline( VMARG )
{
    Item* i_bool = vm->param( 0 );
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
    MYSELF;
    GET_OBJ( self );
    gtk_label_set_use_underline( (GtkLabel*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


FALCON_FUNC Label::set_single_line_mode( VMARG )
{
    Item* i_bool = vm->param( 0 );
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
    MYSELF;
    GET_OBJ( self );
    gtk_label_set_single_line_mode( (GtkLabel*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


FALCON_FUNC Label::set_angle( VMARG )
{
    Item* i_dbl = vm->param( 0 );
    if ( !i_dbl || i_dbl->isNil() || !i_dbl->isOrdinal() )
        throw_inv_params( "O" );
    MYSELF;
    GET_OBJ( self );
    gtk_label_set_angle( (GtkLabel*)_obj, i_dbl->asNumeric() );
}

#if GTK_VERSION_MINOR >= 18
FALCON_FUNC Label::get_current_uri( VMARG )
{
    if ( vm->paramCount() )
        throw_require_no_args();
    MYSELF;
    GET_OBJ( self );
    const gchar* uri = gtk_label_get_current_uri( (GtkLabel*)_obj );
    if ( uri )
        vm->retval( new String( uri ) );
    else
        vm->retnil();
}


FALCON_FUNC Label::set_track_visited_links( VMARG )
{
    Item* i_bool = vm->param( 0 );
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
    MYSELF;
    GET_OBJ( self );
    gtk_label_set_track_visited_links( (GtkLabel*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


FALCON_FUNC Label::get_track_visited_links( VMARG )
{
    if ( vm->paramCount() )
        throw_require_no_args();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_label_get_track_visited_links( (GtkLabel*)_obj ) );
}
#endif


} // Gtk
} // Falcon
