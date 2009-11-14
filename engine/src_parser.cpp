
/* A Bison parser, made by GNU Bison 2.4.1.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C
   
      Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.4.1"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Using locations.  */
#define YYLSP_NEEDED 0

/* Substitute the variable and function names.  */
#define yyparse         flc_src_parse
#define yylex           flc_src_lex
#define yyerror         flc_src_error
#define yylval          flc_src_lval
#define yychar          flc_src_char
#define yydebug         flc_src_debug
#define yynerrs         flc_src_nerrs


/* Copy the first part of user declarations.  */

/* Line 189 of yacc.c  */
#line 17 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"


#include <math.h>
#include <stdio.h>
#include <iostream>

#include <falcon/setup.h>
#include <falcon/compiler.h>
#include <falcon/src_lexer.h>
#include <falcon/syntree.h>
#include <falcon/error.h>
#include <stdlib.h>

#include <falcon/fassert.h>

#include <falcon/memory.h>

#define YYMALLOC	Falcon::memAlloc
#define YYFREE Falcon::memFree

#define  COMPILER  ( reinterpret_cast< Falcon::Compiler *>(yyparam) )
#define  CTX_LINE  ( COMPILER->lexer()->contextStart() )
#define  LINE      ( COMPILER->lexer()->previousLine() )
#define  CURRENT_LINE      ( COMPILER->lexer()->line() )


#define YYPARSE_PARAM yyparam
#define YYLEX_PARAM yyparam

int flc_src_parse( void *param );
void flc_src_error (const char *s);

inline int flc_src_lex (void *lvalp, void *yyparam)
{
   return COMPILER->lexer()->doLex( lvalp );
}

/* Cures a bug in bison 1.8  */
#undef __GNUC_MINOR__



/* Line 189 of yacc.c  */
#line 124 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     EOL = 258,
     INTNUM = 259,
     DBLNUM = 260,
     SYMBOL = 261,
     STRING = 262,
     NIL = 263,
     END = 264,
     DEF = 265,
     WHILE = 266,
     BREAK = 267,
     CONTINUE = 268,
     DROPPING = 269,
     IF = 270,
     ELSE = 271,
     ELIF = 272,
     FOR = 273,
     FORFIRST = 274,
     FORLAST = 275,
     FORMIDDLE = 276,
     SWITCH = 277,
     CASE = 278,
     DEFAULT = 279,
     SELECT = 280,
     SELF = 281,
     FSELF = 282,
     TRY = 283,
     CATCH = 284,
     RAISE = 285,
     CLASS = 286,
     FROM = 287,
     OBJECT = 288,
     RETURN = 289,
     GLOBAL = 290,
     INIT = 291,
     LOAD = 292,
     LAUNCH = 293,
     CONST_KW = 294,
     EXPORT = 295,
     IMPORT = 296,
     DIRECTIVE = 297,
     COLON = 298,
     FUNCDECL = 299,
     STATIC = 300,
     INNERFUNC = 301,
     FORDOT = 302,
     LISTPAR = 303,
     LOOP = 304,
     ENUM = 305,
     TRUE_TOKEN = 306,
     FALSE_TOKEN = 307,
     STATE = 308,
     OUTER_STRING = 309,
     CLOSEPAR = 310,
     OPENPAR = 311,
     CLOSESQUARE = 312,
     OPENSQUARE = 313,
     DOT = 314,
     OPEN_GRAPH = 315,
     CLOSE_GRAPH = 316,
     ARROW = 317,
     VBAR = 318,
     ASSIGN_POW = 319,
     ASSIGN_SHL = 320,
     ASSIGN_SHR = 321,
     ASSIGN_BXOR = 322,
     ASSIGN_BOR = 323,
     ASSIGN_BAND = 324,
     ASSIGN_MOD = 325,
     ASSIGN_DIV = 326,
     ASSIGN_MUL = 327,
     ASSIGN_SUB = 328,
     ASSIGN_ADD = 329,
     OP_EQ = 330,
     OP_AS = 331,
     OP_TO = 332,
     COMMA = 333,
     QUESTION = 334,
     OR = 335,
     AND = 336,
     NOT = 337,
     LE = 338,
     GE = 339,
     LT = 340,
     GT = 341,
     NEQ = 342,
     EEQ = 343,
     PROVIDES = 344,
     OP_NOTIN = 345,
     OP_IN = 346,
     DIESIS = 347,
     ATSIGN = 348,
     CAP_CAP = 349,
     VBAR_VBAR = 350,
     AMPER_AMPER = 351,
     MINUS = 352,
     PLUS = 353,
     PERCENT = 354,
     SLASH = 355,
     STAR = 356,
     POW = 357,
     SHR = 358,
     SHL = 359,
     CAP_XOROOB = 360,
     CAP_ISOOB = 361,
     CAP_DEOOB = 362,
     CAP_OOB = 363,
     CAP_EVAL = 364,
     TILDE = 365,
     NEG = 366,
     AMPER = 367,
     DECREMENT = 368,
     INCREMENT = 369,
     DOLLAR = 370
   };
#endif
/* Tokens.  */
#define EOL 258
#define INTNUM 259
#define DBLNUM 260
#define SYMBOL 261
#define STRING 262
#define NIL 263
#define END 264
#define DEF 265
#define WHILE 266
#define BREAK 267
#define CONTINUE 268
#define DROPPING 269
#define IF 270
#define ELSE 271
#define ELIF 272
#define FOR 273
#define FORFIRST 274
#define FORLAST 275
#define FORMIDDLE 276
#define SWITCH 277
#define CASE 278
#define DEFAULT 279
#define SELECT 280
#define SELF 281
#define FSELF 282
#define TRY 283
#define CATCH 284
#define RAISE 285
#define CLASS 286
#define FROM 287
#define OBJECT 288
#define RETURN 289
#define GLOBAL 290
#define INIT 291
#define LOAD 292
#define LAUNCH 293
#define CONST_KW 294
#define EXPORT 295
#define IMPORT 296
#define DIRECTIVE 297
#define COLON 298
#define FUNCDECL 299
#define STATIC 300
#define INNERFUNC 301
#define FORDOT 302
#define LISTPAR 303
#define LOOP 304
#define ENUM 305
#define TRUE_TOKEN 306
#define FALSE_TOKEN 307
#define STATE 308
#define OUTER_STRING 309
#define CLOSEPAR 310
#define OPENPAR 311
#define CLOSESQUARE 312
#define OPENSQUARE 313
#define DOT 314
#define OPEN_GRAPH 315
#define CLOSE_GRAPH 316
#define ARROW 317
#define VBAR 318
#define ASSIGN_POW 319
#define ASSIGN_SHL 320
#define ASSIGN_SHR 321
#define ASSIGN_BXOR 322
#define ASSIGN_BOR 323
#define ASSIGN_BAND 324
#define ASSIGN_MOD 325
#define ASSIGN_DIV 326
#define ASSIGN_MUL 327
#define ASSIGN_SUB 328
#define ASSIGN_ADD 329
#define OP_EQ 330
#define OP_AS 331
#define OP_TO 332
#define COMMA 333
#define QUESTION 334
#define OR 335
#define AND 336
#define NOT 337
#define LE 338
#define GE 339
#define LT 340
#define GT 341
#define NEQ 342
#define EEQ 343
#define PROVIDES 344
#define OP_NOTIN 345
#define OP_IN 346
#define DIESIS 347
#define ATSIGN 348
#define CAP_CAP 349
#define VBAR_VBAR 350
#define AMPER_AMPER 351
#define MINUS 352
#define PLUS 353
#define PERCENT 354
#define SLASH 355
#define STAR 356
#define POW 357
#define SHR 358
#define SHL 359
#define CAP_XOROOB 360
#define CAP_ISOOB 361
#define CAP_DEOOB 362
#define CAP_OOB 363
#define CAP_EVAL 364
#define TILDE 365
#define NEG 366
#define AMPER 367
#define DECREMENT 368
#define INCREMENT 369
#define DOLLAR 370




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union 
/* Line 214 of yacc.c  */
#line 61 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
lex_value_t
{

/* Line 214 of yacc.c  */
#line 61 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"

   Falcon::int64 integer;
   Falcon::numeric numeric;
   char * charp;
   Falcon::String *stringp;
   Falcon::Symbol *symbol;
   Falcon::Value *fal_val;
   Falcon::Expression *fal_exp;
   Falcon::Statement *fal_stat;
   Falcon::ArrayDecl *fal_adecl;
   Falcon::DictDecl *fal_ddecl;
   Falcon::SymbolList *fal_symlist;
   Falcon::List *fal_genericList;



/* Line 214 of yacc.c  */
#line 410 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 264 of yacc.c  */
#line 422 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int yyi)
#else
static int
YYID (yyi)
    int yyi;
#endif
{
  return yyi;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)				\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack_alloc, Stack, yysize);			\
	Stack = &yyptr->Stack_alloc;					\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   6707

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  116
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  166
/* YYNRULES -- Number of rules.  */
#define YYNRULES  466
/* YYNRULES -- Number of states.  */
#define YYNSTATES  848

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   370

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    11,    14,    18,    20,
      22,    24,    26,    28,    30,    32,    34,    36,    38,    40,
      43,    47,    51,    55,    57,    59,    61,    65,    69,    73,
      76,    79,    84,    91,    93,    95,    97,    99,   101,   103,
     105,   107,   109,   111,   113,   115,   117,   119,   121,   123,
     125,   127,   131,   137,   141,   145,   146,   152,   155,   159,
     163,   167,   171,   172,   180,   184,   188,   189,   191,   192,
     199,   202,   206,   210,   214,   218,   219,   221,   222,   226,
     229,   233,   234,   239,   243,   247,   248,   251,   254,   258,
     261,   265,   269,   270,   277,   278,   285,   291,   295,   298,
     303,   308,   313,   317,   318,   321,   324,   325,   328,   330,
     332,   334,   336,   340,   344,   348,   351,   355,   358,   362,
     366,   368,   369,   376,   380,   384,   385,   392,   396,   400,
     401,   408,   412,   416,   417,   424,   428,   432,   433,   436,
     440,   442,   443,   449,   450,   456,   457,   463,   464,   470,
     471,   472,   476,   477,   479,   482,   485,   488,   490,   494,
     496,   498,   500,   504,   506,   507,   514,   518,   522,   523,
     526,   530,   532,   533,   539,   540,   546,   547,   553,   554,
     560,   562,   566,   567,   569,   571,   575,   576,   583,   586,
     590,   591,   593,   595,   598,   601,   604,   609,   613,   619,
     623,   625,   629,   631,   633,   637,   641,   647,   650,   656,
     657,   665,   669,   675,   676,   683,   686,   687,   689,   693,
     695,   696,   697,   703,   704,   708,   711,   715,   718,   722,
     726,   730,   736,   742,   746,   749,   753,   757,   759,   763,
     767,   773,   779,   787,   795,   803,   811,   816,   821,   826,
     831,   838,   845,   849,   854,   859,   861,   865,   869,   873,
     875,   879,   883,   887,   891,   892,   900,   904,   907,   908,
     912,   913,   919,   920,   923,   925,   929,   932,   933,   936,
     940,   941,   944,   946,   948,   950,   952,   954,   956,   957,
     965,   971,   976,   981,   985,   989,   990,   993,   995,   997,
     998,  1006,  1007,  1010,  1012,  1017,  1019,  1022,  1024,  1026,
    1027,  1035,  1038,  1041,  1042,  1045,  1047,  1049,  1051,  1053,
    1055,  1056,  1061,  1063,  1065,  1068,  1072,  1076,  1078,  1081,
    1085,  1089,  1091,  1093,  1095,  1097,  1099,  1101,  1103,  1105,
    1107,  1109,  1111,  1113,  1115,  1117,  1119,  1121,  1122,  1124,
    1126,  1128,  1131,  1134,  1137,  1141,  1145,  1149,  1152,  1156,
    1161,  1166,  1171,  1176,  1181,  1186,  1191,  1196,  1201,  1206,
    1211,  1214,  1218,  1221,  1224,  1227,  1230,  1234,  1238,  1242,
    1246,  1250,  1254,  1258,  1261,  1265,  1269,  1273,  1276,  1279,
    1282,  1285,  1288,  1291,  1294,  1297,  1300,  1302,  1304,  1306,
    1308,  1310,  1312,  1315,  1317,  1322,  1328,  1332,  1334,  1336,
    1340,  1346,  1350,  1354,  1358,  1362,  1366,  1370,  1374,  1378,
    1382,  1386,  1390,  1394,  1398,  1403,  1408,  1414,  1422,  1427,
    1431,  1432,  1439,  1440,  1447,  1448,  1455,  1460,  1464,  1467,
    1470,  1473,  1476,  1477,  1484,  1490,  1496,  1501,  1505,  1508,
    1512,  1516,  1519,  1523,  1527,  1531,  1535,  1540,  1542,  1546,
    1548,  1552,  1553,  1555,  1557,  1561,  1565
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     117,     0,    -1,   118,    -1,    -1,   118,   119,    -1,   120,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   122,    -1,
     220,    -1,   200,    -1,   223,    -1,   246,    -1,   241,    -1,
     123,    -1,   214,    -1,   215,    -1,   217,    -1,     4,    -1,
      97,     4,    -1,    37,     6,     3,    -1,    37,     7,     3,
      -1,    37,     1,     3,    -1,   124,    -1,   218,    -1,     3,
      -1,    44,     1,     3,    -1,    33,     1,     3,    -1,    31,
       1,     3,    -1,     1,     3,    -1,   261,     3,    -1,   277,
      75,   261,     3,    -1,   277,    75,   261,    78,   277,     3,
      -1,   126,    -1,   127,    -1,   131,    -1,   147,    -1,   164,
      -1,   179,    -1,   134,    -1,   145,    -1,   146,    -1,   190,
      -1,   199,    -1,   255,    -1,   251,    -1,   213,    -1,   155,
      -1,   156,    -1,   157,    -1,   258,    -1,   258,    75,   261,
      -1,   125,    78,   258,    75,   261,    -1,    10,   125,     3,
      -1,    10,     1,     3,    -1,    -1,   129,   128,   144,     9,
       3,    -1,   130,   123,    -1,    11,   261,     3,    -1,    11,
       1,     3,    -1,    11,   261,    43,    -1,    11,     1,    43,
      -1,    -1,    49,     3,   132,   144,     9,   133,     3,    -1,
      49,    43,   123,    -1,    49,     1,     3,    -1,    -1,   261,
      -1,    -1,   136,   135,   144,   138,     9,     3,    -1,   137,
     123,    -1,    15,   261,     3,    -1,    15,     1,     3,    -1,
      15,   261,    43,    -1,    15,     1,    43,    -1,    -1,   141,
      -1,    -1,   140,   139,   144,    -1,    16,     3,    -1,    16,
       1,     3,    -1,    -1,   143,   142,   144,   138,    -1,    17,
     261,     3,    -1,    17,     1,     3,    -1,    -1,   144,   123,
      -1,    12,     3,    -1,    12,     1,     3,    -1,    13,     3,
      -1,    13,    14,     3,    -1,    13,     1,     3,    -1,    -1,
      18,   280,    91,   261,   148,   150,    -1,    -1,    18,   258,
      75,   151,   149,   150,    -1,    18,   280,    91,     1,     3,
      -1,    18,     1,     3,    -1,    43,   123,    -1,     3,   153,
       9,     3,    -1,   261,    77,   261,   152,    -1,   261,    77,
     261,     1,    -1,   261,    77,     1,    -1,    -1,    78,   261,
      -1,    78,     1,    -1,    -1,   154,   153,    -1,   123,    -1,
     158,    -1,   160,    -1,   162,    -1,    47,   261,     3,    -1,
      47,     1,     3,    -1,   103,   277,     3,    -1,   103,     3,
      -1,    86,   277,     3,    -1,    86,     3,    -1,   103,     1,
       3,    -1,    86,     1,     3,    -1,    54,    -1,    -1,    19,
       3,   159,   144,     9,     3,    -1,    19,    43,   123,    -1,
      19,     1,     3,    -1,    -1,    20,     3,   161,   144,     9,
       3,    -1,    20,    43,   123,    -1,    20,     1,     3,    -1,
      -1,    21,     3,   163,   144,     9,     3,    -1,    21,    43,
     123,    -1,    21,     1,     3,    -1,    -1,   166,   165,   167,
     173,     9,     3,    -1,    22,   261,     3,    -1,    22,     1,
       3,    -1,    -1,   167,   168,    -1,   167,     1,     3,    -1,
       3,    -1,    -1,    23,   177,     3,   169,   144,    -1,    -1,
      23,   177,    43,   170,   123,    -1,    -1,    23,     1,     3,
     171,   144,    -1,    -1,    23,     1,    43,   172,   123,    -1,
      -1,    -1,   175,   174,   176,    -1,    -1,    24,    -1,    24,
       1,    -1,     3,   144,    -1,    43,   123,    -1,   178,    -1,
     177,    78,   178,    -1,     8,    -1,   121,    -1,     7,    -1,
     121,    77,   121,    -1,     6,    -1,    -1,   181,   180,   182,
     173,     9,     3,    -1,    25,   261,     3,    -1,    25,     1,
       3,    -1,    -1,   182,   183,    -1,   182,     1,     3,    -1,
       3,    -1,    -1,    23,   188,     3,   184,   144,    -1,    -1,
      23,   188,    43,   185,   123,    -1,    -1,    23,     1,     3,
     186,   144,    -1,    -1,    23,     1,    43,   187,   123,    -1,
     189,    -1,   188,    78,   189,    -1,    -1,     4,    -1,     6,
      -1,    28,    43,   123,    -1,    -1,   192,   191,   144,   193,
       9,     3,    -1,    28,     3,    -1,    28,     1,     3,    -1,
      -1,   194,    -1,   195,    -1,   194,   195,    -1,   196,   144,
      -1,    29,     3,    -1,    29,    91,   258,     3,    -1,    29,
     197,     3,    -1,    29,   197,    91,   258,     3,    -1,    29,
       1,     3,    -1,   198,    -1,   197,    78,   198,    -1,     4,
      -1,     6,    -1,    30,   261,     3,    -1,    30,     1,     3,
      -1,   201,   208,   144,     9,     3,    -1,   203,   123,    -1,
     205,    56,   206,    55,     3,    -1,    -1,   205,    56,   206,
       1,   202,    55,     3,    -1,   205,     1,     3,    -1,   205,
      56,   206,    55,    43,    -1,    -1,   205,    56,     1,   204,
      55,    43,    -1,    44,     6,    -1,    -1,   207,    -1,   206,
      78,   207,    -1,     6,    -1,    -1,    -1,   211,   209,   144,
       9,     3,    -1,    -1,   212,   210,   123,    -1,    45,     3,
      -1,    45,     1,     3,    -1,    45,    43,    -1,    45,     1,
      43,    -1,    38,   263,     3,    -1,    38,     1,     3,    -1,
      39,     6,    75,   257,     3,    -1,    39,     6,    75,     1,
       3,    -1,    39,     1,     3,    -1,    40,     3,    -1,    40,
     216,     3,    -1,    40,     1,     3,    -1,     6,    -1,   216,
      78,     6,    -1,    41,   219,     3,    -1,    41,   219,    32,
       6,     3,    -1,    41,   219,    32,     7,     3,    -1,    41,
     219,    32,     6,    76,     6,     3,    -1,    41,   219,    32,
       7,    76,     6,     3,    -1,    41,   219,    32,     6,    91,
       6,     3,    -1,    41,   219,    32,     7,    91,     6,     3,
      -1,    41,     6,     1,     3,    -1,    41,   219,     1,     3,
      -1,    41,    32,     6,     3,    -1,    41,    32,     7,     3,
      -1,    41,    32,     6,    76,     6,     3,    -1,    41,    32,
       7,    76,     6,     3,    -1,    41,     1,     3,    -1,     6,
      43,   257,     3,    -1,     6,    43,     1,     3,    -1,     6,
      -1,   219,    78,     6,    -1,    42,   221,     3,    -1,    42,
       1,     3,    -1,   222,    -1,   221,    78,   222,    -1,     6,
      75,     6,    -1,     6,    75,     7,    -1,     6,    75,   121,
      -1,    -1,    31,     6,   224,   225,   232,     9,     3,    -1,
     226,   228,     3,    -1,     1,     3,    -1,    -1,    56,   206,
      55,    -1,    -1,    56,   206,     1,   227,    55,    -1,    -1,
      32,   229,    -1,   230,    -1,   229,    78,   230,    -1,     6,
     231,    -1,    -1,    56,    55,    -1,    56,   277,    55,    -1,
      -1,   232,   233,    -1,     3,    -1,   200,    -1,   236,    -1,
     237,    -1,   234,    -1,   218,    -1,    -1,    36,     3,   235,
     208,   144,     9,     3,    -1,    45,     6,    75,   257,     3,
      -1,     6,    75,   261,     3,    -1,   238,   239,     9,     3,
      -1,    53,     6,     3,    -1,    53,    36,     3,    -1,    -1,
     239,   240,    -1,     3,    -1,   200,    -1,    -1,    50,     6,
     242,     3,   243,     9,     3,    -1,    -1,   243,   244,    -1,
       3,    -1,     6,    75,   257,   245,    -1,   218,    -1,     6,
     245,    -1,     3,    -1,    78,    -1,    -1,    33,     6,   247,
     248,   249,     9,     3,    -1,   228,     3,    -1,     1,     3,
      -1,    -1,   249,   250,    -1,     3,    -1,   200,    -1,   236,
      -1,   234,    -1,   218,    -1,    -1,    35,   252,   253,     3,
      -1,   254,    -1,     1,    -1,   254,     1,    -1,   253,    78,
     254,    -1,   253,    78,     1,    -1,     6,    -1,    34,     3,
      -1,    34,   261,     3,    -1,    34,     1,     3,    -1,     8,
      -1,    51,    -1,    52,    -1,     4,    -1,     5,    -1,     7,
      -1,     8,    -1,    51,    -1,    52,    -1,   121,    -1,     5,
      -1,     7,    -1,     6,    -1,   258,    -1,    26,    -1,    27,
      -1,    -1,     3,    -1,   256,    -1,   259,    -1,   112,     6,
      -1,   112,     4,    -1,   112,    26,    -1,   112,    59,     6,
      -1,   112,    59,     4,    -1,   112,    59,    26,    -1,    97,
     261,    -1,     6,    63,   261,    -1,   261,    98,   260,   261,
      -1,   261,    97,   260,   261,    -1,   261,   101,   260,   261,
      -1,   261,   100,   260,   261,    -1,   261,    99,   260,   261,
      -1,   261,   102,   260,   261,    -1,   261,    96,   260,   261,
      -1,   261,    95,   260,   261,    -1,   261,    94,   260,   261,
      -1,   261,   104,   260,   261,    -1,   261,   103,   260,   261,
      -1,   110,   261,    -1,   261,    87,   261,    -1,   261,   114,
      -1,   114,   261,    -1,   261,   113,    -1,   113,   261,    -1,
     261,    88,   261,    -1,   261,    86,   261,    -1,   261,    85,
     261,    -1,   261,    84,   261,    -1,   261,    83,   261,    -1,
     261,    81,   261,    -1,   261,    80,   261,    -1,    82,   261,
      -1,   261,    91,   261,    -1,   261,    90,   261,    -1,   261,
      89,     6,    -1,   115,   258,    -1,   115,     4,    -1,    93,
     261,    -1,    92,   261,    -1,   109,   261,    -1,   108,   261,
      -1,   107,   261,    -1,   106,   261,    -1,   105,   261,    -1,
     265,    -1,   267,    -1,   271,    -1,   263,    -1,   273,    -1,
     275,    -1,   261,   262,    -1,   274,    -1,   261,    58,   261,
      57,    -1,   261,    58,   101,   261,    57,    -1,   261,    59,
       6,    -1,   276,    -1,   262,    -1,   261,    75,   261,    -1,
     261,    75,   261,    78,   277,    -1,   261,    74,   261,    -1,
     261,    73,   261,    -1,   261,    72,   261,    -1,   261,    71,
     261,    -1,   261,    70,   261,    -1,   261,    64,   261,    -1,
     261,    69,   261,    -1,   261,    68,   261,    -1,   261,    67,
     261,    -1,   261,    65,   261,    -1,   261,    66,   261,    -1,
      56,   261,    55,    -1,    58,    43,    57,    -1,    58,   261,
      43,    57,    -1,    58,    43,   261,    57,    -1,    58,   261,
      43,   261,    57,    -1,    58,   261,    43,   261,    43,   261,
      57,    -1,   261,    56,   277,    55,    -1,   261,    56,    55,
      -1,    -1,   261,    56,   277,     1,   264,    55,    -1,    -1,
      44,   266,   269,   208,   144,     9,    -1,    -1,    60,   268,
     270,   208,   144,    61,    -1,    56,   206,    55,     3,    -1,
      56,   206,     1,    -1,     1,     3,    -1,   206,    62,    -1,
     206,     1,    -1,     1,    62,    -1,    -1,    46,   272,   269,
     208,   144,     9,    -1,   261,    79,   261,    43,   261,    -1,
     261,    79,   261,    43,     1,    -1,   261,    79,   261,     1,
      -1,   261,    79,     1,    -1,    58,    57,    -1,    58,   277,
      57,    -1,    58,   277,     1,    -1,    48,    57,    -1,    48,
     278,    57,    -1,    48,   278,     1,    -1,    58,    62,    57,
      -1,    58,   281,    57,    -1,    58,   281,     1,    57,    -1,
     261,    -1,   277,    78,   261,    -1,   261,    -1,   278,   279,
     261,    -1,    -1,    78,    -1,   258,    -1,   280,    78,   258,
      -1,   261,    62,   261,    -1,   281,    78,   261,    62,   261,
      -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   197,   197,   200,   202,   206,   207,   208,   212,   213,
     214,   219,   224,   229,   234,   239,   240,   241,   245,   246,
     250,   256,   262,   269,   270,   271,   272,   273,   274,   275,
     280,   291,   297,   311,   312,   313,   314,   315,   316,   317,
     318,   319,   320,   321,   322,   323,   324,   325,   326,   327,
     331,   334,   340,   348,   350,   355,   355,   369,   377,   378,
     382,   383,   387,   387,   402,   408,   415,   416,   420,   420,
     435,   445,   446,   450,   451,   455,   457,   458,   458,   467,
     468,   473,   473,   485,   486,   489,   491,   497,   506,   514,
     524,   533,   543,   542,   563,   562,   585,   590,   598,   604,
     611,   617,   621,   628,   629,   630,   633,   635,   639,   646,
     647,   648,   652,   665,   673,   677,   683,   689,   696,   701,
     710,   720,   720,   734,   743,   747,   747,   760,   769,   773,
     773,   789,   798,   802,   802,   819,   820,   827,   829,   830,
     834,   836,   835,   846,   846,   858,   858,   870,   870,   886,
     889,   888,   901,   902,   903,   906,   907,   913,   914,   918,
     927,   939,   950,   961,   982,   982,   999,  1000,  1007,  1009,
    1010,  1014,  1016,  1015,  1026,  1026,  1039,  1039,  1051,  1051,
    1069,  1070,  1073,  1074,  1086,  1107,  1114,  1113,  1132,  1133,
    1136,  1138,  1142,  1143,  1147,  1152,  1170,  1190,  1200,  1211,
    1219,  1220,  1224,  1236,  1259,  1260,  1267,  1277,  1286,  1287,
    1287,  1291,  1295,  1296,  1296,  1303,  1402,  1404,  1405,  1409,
    1424,  1427,  1426,  1438,  1437,  1452,  1453,  1457,  1458,  1467,
    1471,  1479,  1489,  1494,  1506,  1515,  1522,  1530,  1535,  1543,
    1548,  1553,  1558,  1578,  1597,  1602,  1607,  1612,  1626,  1631,
    1636,  1641,  1646,  1655,  1660,  1667,  1673,  1685,  1690,  1698,
    1699,  1703,  1707,  1711,  1725,  1724,  1787,  1790,  1796,  1798,
    1799,  1799,  1805,  1807,  1811,  1812,  1816,  1840,  1841,  1842,
    1849,  1851,  1855,  1856,  1859,  1878,  1890,  1891,  1895,  1895,
    1929,  1951,  1979,  1991,  1997,  2010,  2012,  2017,  2018,  2030,
    2029,  2073,  2075,  2079,  2080,  2084,  2085,  2092,  2092,  2101,
    2100,  2167,  2168,  2174,  2176,  2180,  2181,  2184,  2203,  2204,
    2213,  2212,  2230,  2231,  2236,  2241,  2242,  2249,  2265,  2266,
    2267,  2275,  2276,  2277,  2278,  2279,  2280,  2284,  2285,  2286,
    2287,  2288,  2289,  2293,  2311,  2312,  2313,  2333,  2335,  2339,
    2340,  2341,  2342,  2343,  2344,  2345,  2346,  2347,  2348,  2349,
    2375,  2376,  2396,  2420,  2437,  2438,  2439,  2440,  2441,  2442,
    2443,  2444,  2445,  2446,  2447,  2448,  2449,  2450,  2451,  2452,
    2453,  2454,  2455,  2456,  2457,  2458,  2459,  2460,  2461,  2462,
    2463,  2464,  2465,  2466,  2467,  2468,  2469,  2470,  2471,  2472,
    2473,  2474,  2476,  2481,  2485,  2490,  2496,  2505,  2506,  2508,
    2513,  2520,  2521,  2522,  2523,  2524,  2525,  2526,  2527,  2528,
    2529,  2530,  2531,  2536,  2539,  2542,  2545,  2548,  2554,  2560,
    2565,  2565,  2575,  2574,  2618,  2617,  2669,  2670,  2674,  2681,
    2682,  2686,  2694,  2693,  2743,  2748,  2755,  2762,  2772,  2773,
    2777,  2785,  2786,  2790,  2799,  2800,  2801,  2809,  2810,  2814,
    2815,  2818,  2819,  2822,  2828,  2835,  2836
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "EOL", "INTNUM", "DBLNUM", "SYMBOL",
  "STRING", "NIL", "END", "DEF", "WHILE", "BREAK", "CONTINUE", "DROPPING",
  "IF", "ELSE", "ELIF", "FOR", "FORFIRST", "FORLAST", "FORMIDDLE",
  "SWITCH", "CASE", "DEFAULT", "SELECT", "SELF", "FSELF", "TRY", "CATCH",
  "RAISE", "CLASS", "FROM", "OBJECT", "RETURN", "GLOBAL", "INIT", "LOAD",
  "LAUNCH", "CONST_KW", "EXPORT", "IMPORT", "DIRECTIVE", "COLON",
  "FUNCDECL", "STATIC", "INNERFUNC", "FORDOT", "LISTPAR", "LOOP", "ENUM",
  "TRUE_TOKEN", "FALSE_TOKEN", "STATE", "OUTER_STRING", "CLOSEPAR",
  "OPENPAR", "CLOSESQUARE", "OPENSQUARE", "DOT", "OPEN_GRAPH",
  "CLOSE_GRAPH", "ARROW", "VBAR", "ASSIGN_POW", "ASSIGN_SHL", "ASSIGN_SHR",
  "ASSIGN_BXOR", "ASSIGN_BOR", "ASSIGN_BAND", "ASSIGN_MOD", "ASSIGN_DIV",
  "ASSIGN_MUL", "ASSIGN_SUB", "ASSIGN_ADD", "OP_EQ", "OP_AS", "OP_TO",
  "COMMA", "QUESTION", "OR", "AND", "NOT", "LE", "GE", "LT", "GT", "NEQ",
  "EEQ", "PROVIDES", "OP_NOTIN", "OP_IN", "DIESIS", "ATSIGN", "CAP_CAP",
  "VBAR_VBAR", "AMPER_AMPER", "MINUS", "PLUS", "PERCENT", "SLASH", "STAR",
  "POW", "SHR", "SHL", "CAP_XOROOB", "CAP_ISOOB", "CAP_DEOOB", "CAP_OOB",
  "CAP_EVAL", "TILDE", "NEG", "AMPER", "DECREMENT", "INCREMENT", "DOLLAR",
  "$accept", "input", "body", "line", "toplevel_statement",
  "INTNUM_WITH_MINUS", "load_statement", "statement", "base_statement",
  "assignment_def_list", "def_statement", "while_statement", "$@1",
  "while_decl", "while_short_decl", "loop_statement", "$@2",
  "loop_terminator", "if_statement", "$@3", "if_decl", "if_short_decl",
  "elif_or_else", "$@4", "else_decl", "elif_statement", "$@5", "elif_decl",
  "statement_list", "break_statement", "continue_statement",
  "forin_statement", "$@6", "$@7", "forin_rest", "for_to_expr",
  "for_to_step_clause", "forin_statement_list", "forin_statement_elem",
  "fordot_statement", "self_print_statement", "outer_print_statement",
  "first_loop_block", "$@8", "last_loop_block", "$@9", "middle_loop_block",
  "$@10", "switch_statement", "$@11", "switch_decl", "case_list",
  "case_statement", "$@12", "$@13", "$@14", "$@15", "default_statement",
  "$@16", "default_decl", "default_body", "case_expression_list",
  "case_element", "select_statement", "$@17", "select_decl",
  "selcase_list", "selcase_statement", "$@18", "$@19", "$@20", "$@21",
  "selcase_expression_list", "selcase_element", "try_statement", "$@22",
  "try_decl", "catch_statements", "catch_list", "catch_statement",
  "catch_decl", "catchcase_element_list", "catchcase_element",
  "raise_statement", "func_statement", "func_decl", "$@23",
  "func_decl_short", "$@24", "func_begin", "param_list", "param_symbol",
  "static_block", "$@25", "$@26", "static_decl", "static_short_decl",
  "launch_statement", "const_statement", "export_statement",
  "export_symbol_list", "import_statement", "attribute_statement",
  "import_symbol_list", "directive_statement", "directive_pair_list",
  "directive_pair", "class_decl", "$@27", "class_def_inner",
  "class_param_list", "$@28", "from_clause", "inherit_list",
  "inherit_token", "inherit_call", "class_statement_list",
  "class_statement", "init_decl", "$@29", "property_decl", "state_decl",
  "state_heading", "state_statement_list", "state_statement",
  "enum_statement", "$@30", "enum_statement_list", "enum_item_decl",
  "enum_item_terminator", "object_decl", "$@31", "object_decl_inner",
  "object_statement_list", "object_statement", "global_statement", "$@32",
  "global_symbol_list", "globalized_symbol", "return_statement",
  "const_atom_non_minus", "const_atom", "atomic_symbol", "var_atom",
  "OPT_EOL", "expression", "range_decl", "func_call", "$@33",
  "nameless_func", "$@34", "nameless_block", "$@35",
  "nameless_func_decl_inner", "nameless_block_decl_inner", "innerfunc",
  "$@36", "iif_expr", "array_decl", "dotarray_decl", "dict_decl",
  "expression_list", "listpar_expression_list", "listpar_comma",
  "symbol_list", "expression_pair_list", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   116,   117,   118,   118,   119,   119,   119,   120,   120,
     120,   120,   120,   120,   120,   120,   120,   120,   121,   121,
     122,   122,   122,   123,   123,   123,   123,   123,   123,   123,
     124,   124,   124,   124,   124,   124,   124,   124,   124,   124,
     124,   124,   124,   124,   124,   124,   124,   124,   124,   124,
     125,   125,   125,   126,   126,   128,   127,   127,   129,   129,
     130,   130,   132,   131,   131,   131,   133,   133,   135,   134,
     134,   136,   136,   137,   137,   138,   138,   139,   138,   140,
     140,   142,   141,   143,   143,   144,   144,   145,   145,   146,
     146,   146,   148,   147,   149,   147,   147,   147,   150,   150,
     151,   151,   151,   152,   152,   152,   153,   153,   154,   154,
     154,   154,   155,   155,   156,   156,   156,   156,   156,   156,
     157,   159,   158,   158,   158,   161,   160,   160,   160,   163,
     162,   162,   162,   165,   164,   166,   166,   167,   167,   167,
     168,   169,   168,   170,   168,   171,   168,   172,   168,   173,
     174,   173,   175,   175,   175,   176,   176,   177,   177,   178,
     178,   178,   178,   178,   180,   179,   181,   181,   182,   182,
     182,   183,   184,   183,   185,   183,   186,   183,   187,   183,
     188,   188,   189,   189,   189,   190,   191,   190,   192,   192,
     193,   193,   194,   194,   195,   196,   196,   196,   196,   196,
     197,   197,   198,   198,   199,   199,   200,   200,   201,   202,
     201,   201,   203,   204,   203,   205,   206,   206,   206,   207,
     208,   209,   208,   210,   208,   211,   211,   212,   212,   213,
     213,   214,   214,   214,   215,   215,   215,   216,   216,   217,
     217,   217,   217,   217,   217,   217,   217,   217,   217,   217,
     217,   217,   217,   218,   218,   219,   219,   220,   220,   221,
     221,   222,   222,   222,   224,   223,   225,   225,   226,   226,
     227,   226,   228,   228,   229,   229,   230,   231,   231,   231,
     232,   232,   233,   233,   233,   233,   233,   233,   235,   234,
     236,   236,   237,   238,   238,   239,   239,   240,   240,   242,
     241,   243,   243,   244,   244,   244,   244,   245,   245,   247,
     246,   248,   248,   249,   249,   250,   250,   250,   250,   250,
     252,   251,   253,   253,   253,   253,   253,   254,   255,   255,
     255,   256,   256,   256,   256,   256,   256,   257,   257,   257,
     257,   257,   257,   258,   259,   259,   259,   260,   260,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   262,   262,   262,   262,   262,   263,   263,
     264,   263,   266,   265,   268,   267,   269,   269,   269,   270,
     270,   270,   272,   271,   273,   273,   273,   273,   274,   274,
     274,   275,   275,   275,   276,   276,   276,   277,   277,   278,
     278,   279,   279,   280,   280,   281,   281
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       3,     3,     3,     1,     1,     1,     3,     3,     3,     2,
       2,     4,     6,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     5,     3,     3,     0,     5,     2,     3,     3,
       3,     3,     0,     7,     3,     3,     0,     1,     0,     6,
       2,     3,     3,     3,     3,     0,     1,     0,     3,     2,
       3,     0,     4,     3,     3,     0,     2,     2,     3,     2,
       3,     3,     0,     6,     0,     6,     5,     3,     2,     4,
       4,     4,     3,     0,     2,     2,     0,     2,     1,     1,
       1,     1,     3,     3,     3,     2,     3,     2,     3,     3,
       1,     0,     6,     3,     3,     0,     6,     3,     3,     0,
       6,     3,     3,     0,     6,     3,     3,     0,     2,     3,
       1,     0,     5,     0,     5,     0,     5,     0,     5,     0,
       0,     3,     0,     1,     2,     2,     2,     1,     3,     1,
       1,     1,     3,     1,     0,     6,     3,     3,     0,     2,
       3,     1,     0,     5,     0,     5,     0,     5,     0,     5,
       1,     3,     0,     1,     1,     3,     0,     6,     2,     3,
       0,     1,     1,     2,     2,     2,     4,     3,     5,     3,
       1,     3,     1,     1,     3,     3,     5,     2,     5,     0,
       7,     3,     5,     0,     6,     2,     0,     1,     3,     1,
       0,     0,     5,     0,     3,     2,     3,     2,     3,     3,
       3,     5,     5,     3,     2,     3,     3,     1,     3,     3,
       5,     5,     7,     7,     7,     7,     4,     4,     4,     4,
       6,     6,     3,     4,     4,     1,     3,     3,     3,     1,
       3,     3,     3,     3,     0,     7,     3,     2,     0,     3,
       0,     5,     0,     2,     1,     3,     2,     0,     2,     3,
       0,     2,     1,     1,     1,     1,     1,     1,     0,     7,
       5,     4,     4,     3,     3,     0,     2,     1,     1,     0,
       7,     0,     2,     1,     4,     1,     2,     1,     1,     0,
       7,     2,     2,     0,     2,     1,     1,     1,     1,     1,
       0,     4,     1,     1,     2,     3,     3,     1,     2,     3,
       3,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     0,     1,     1,
       1,     2,     2,     2,     3,     3,     3,     2,     3,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     4,     4,
       2,     3,     2,     2,     2,     2,     3,     3,     3,     3,
       3,     3,     3,     2,     3,     3,     3,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     1,     1,     1,
       1,     1,     2,     1,     4,     5,     3,     1,     1,     3,
       5,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     4,     4,     5,     7,     4,     3,
       0,     6,     0,     6,     0,     6,     4,     3,     2,     2,
       2,     2,     0,     6,     5,     5,     4,     3,     2,     3,
       3,     2,     3,     3,     3,     3,     4,     1,     3,     1,
       3,     0,     1,     1,     3,     3,     5
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    25,   334,   335,   343,   336,
     331,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   345,   346,     0,     0,     0,     0,     0,   320,     0,
       0,     0,     0,     0,     0,     0,   442,     0,     0,     0,
       0,   332,   333,   120,     0,     0,   434,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     4,     5,     8,    14,    23,    33,    34,
      55,     0,    35,    39,    68,     0,    40,    41,    36,    47,
      48,    49,    37,   133,    38,   164,    42,   186,    43,    10,
     220,     0,     0,    46,    15,    16,    17,    24,     9,    11,
      13,    12,    45,    44,   349,   344,   350,   457,   408,   399,
     396,   397,   398,   400,   403,   401,   407,     0,    29,     0,
       0,     6,     0,   343,     0,    50,     0,   343,   432,     0,
       0,    87,     0,    89,     0,     0,     0,     0,   463,     0,
       0,     0,     0,     0,     0,     0,   188,     0,     0,     0,
       0,   264,     0,   309,     0,   328,     0,     0,     0,     0,
       0,     0,     0,   399,     0,     0,     0,   234,   237,     0,
       0,     0,     0,     0,     0,     0,     0,   259,     0,   215,
       0,     0,     0,     0,   451,   459,     0,     0,    62,     0,
     299,     0,     0,   448,     0,   457,     0,     0,     0,   383,
       0,   117,   457,     0,   390,   389,   357,     0,   115,     0,
     395,   394,   393,   392,   391,   370,   352,   351,   353,     0,
     375,   373,   388,   387,    85,     0,     0,     0,    57,    85,
      70,   137,   168,    85,     0,    85,   221,   223,   207,     0,
       0,    30,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   347,
     347,   347,   347,   347,   347,   347,   347,   347,   347,   347,
     374,   372,   402,     0,     0,     0,    18,   341,   342,   337,
     338,   339,     0,   340,     0,   358,    54,    53,     0,     0,
      59,    61,    58,    60,    88,    91,    90,    72,    74,    71,
      73,    97,     0,     0,     0,   136,   135,     7,   167,   166,
     189,   185,   205,   204,    28,     0,    27,     0,   330,   329,
     323,   327,     0,     0,    22,    20,    21,   230,   229,   233,
       0,   236,   235,     0,   252,     0,     0,     0,     0,   239,
       0,     0,   258,     0,   257,     0,    26,     0,   216,   220,
     220,   113,   112,   453,   452,   462,     0,    65,    85,    64,
       0,   422,   423,     0,   454,     0,     0,   450,   449,     0,
     455,     0,     0,   219,     0,   217,   220,   119,   116,   118,
     114,   355,   354,   356,     0,     0,     0,     0,     0,     0,
     225,   227,     0,    85,     0,   211,   213,     0,   429,     0,
       0,     0,   406,   416,   420,   421,   419,   418,   417,   415,
     414,   413,   412,   411,   409,   447,     0,   382,   381,   380,
     379,   378,   377,   371,   376,   386,   385,   384,   348,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   458,   254,    19,   253,     0,    51,    94,     0,   464,
       0,    92,     0,   216,   280,   272,     0,     0,     0,   313,
     321,     0,   324,     0,     0,   238,   246,   248,     0,   249,
       0,   247,     0,     0,   256,   261,   262,   263,   260,   438,
       0,    85,    85,   460,     0,   301,   425,   424,     0,   465,
     456,     0,   441,   440,   439,     0,    85,     0,    86,     0,
       0,     0,    77,    76,    81,     0,   140,     0,     0,   138,
       0,   150,     0,   171,     0,     0,   169,     0,     0,   191,
     192,    85,   226,   228,     0,     0,   224,     0,   209,     0,
     430,   428,     0,   404,     0,   446,     0,   367,   366,   365,
     360,   359,   363,   362,   361,   364,   369,   368,    31,     0,
       0,     0,     0,    96,     0,   267,     0,     0,     0,   312,
     277,   273,   274,   311,     0,   326,   325,   232,   231,     0,
       0,   240,     0,     0,   241,     0,     0,   437,     0,     0,
       0,    66,     0,     0,   426,     0,   218,     0,    56,     0,
      79,     0,     0,     0,    85,    85,   139,     0,   163,   161,
     159,   160,     0,   157,   154,     0,     0,   170,     0,   183,
     184,     0,   180,     0,     0,   195,   202,   203,     0,     0,
     200,     0,   193,     0,   206,     0,     0,     0,   208,   212,
       0,   405,   410,   445,   444,     0,    52,     0,     0,    95,
     102,     0,    93,   270,   269,   282,     0,     0,     0,     0,
       0,     0,   283,   287,   281,   286,   284,   285,   295,   266,
       0,   276,     0,   315,     0,   316,   319,   318,   317,   314,
     250,   251,     0,     0,     0,     0,   436,   433,   443,     0,
      67,   303,     0,     0,   305,   302,     0,   466,   435,    80,
      84,    83,    69,     0,     0,   145,   147,     0,   141,   143,
       0,   134,    85,     0,   151,   176,   178,   172,   174,   182,
     165,   199,     0,   197,     0,     0,   187,   222,   214,     0,
     431,    32,     0,     0,     0,   108,     0,     0,   109,   110,
     111,    98,   101,     0,   100,     0,     0,   265,   288,     0,
       0,     0,     0,   278,     0,   275,   310,   242,   244,   243,
     245,    63,   307,     0,   308,   306,   300,   427,    82,    85,
       0,   162,    85,     0,   158,     0,   156,    85,     0,    85,
       0,   181,   196,   201,     0,   210,     0,   121,     0,     0,
     125,     0,     0,   129,     0,     0,   107,   105,   104,   271,
       0,   220,     0,   293,   294,   297,     0,   298,   296,   279,
       0,     0,   148,     0,   144,     0,   179,     0,   175,   198,
     124,    85,   123,   128,    85,   127,   132,    85,   131,    99,
     291,    85,     0,   292,   304,     0,     0,     0,     0,   290,
       0,     0,     0,     0,   122,   126,   130,   289
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    63,    64,   293,    65,   508,    67,   124,
      68,    69,   224,    70,    71,    72,   368,   689,    73,   229,
      74,    75,   511,   604,   512,   513,   605,   514,   394,    76,
      77,    78,   564,   561,   649,   457,   744,   736,   737,    79,
      80,    81,   738,   821,   739,   824,   740,   827,    82,   231,
      83,   396,   519,   772,   773,   769,   770,   520,   616,   521,
     714,   612,   613,    84,   232,    85,   397,   526,   779,   780,
     777,   778,   621,   622,    86,   233,    87,   528,   529,   530,
     531,   629,   630,    88,    89,    90,   637,    91,   537,    92,
     384,   385,   235,   403,   404,   236,   237,    93,    94,    95,
     169,    96,    97,   173,    98,   176,   177,    99,   325,   464,
     465,   745,   468,   571,   572,   671,   567,   664,   665,   801,
     666,   667,   668,   752,   808,   100,   370,   592,   695,   765,
     101,   327,   469,   574,   679,   102,   157,   332,   333,   103,
     104,   294,   105,   106,   439,   107,   108,   109,   640,   110,
     180,   111,   198,   359,   386,   112,   181,   113,   114,   115,
     116,   117,   186,   366,   139,   197
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -559
static const yytype_int16 yypact[] =
{
    -559,    23,   790,  -559,    21,  -559,  -559,  -559,   410,  -559,
    -559,    47,    13,  3608,   352,   357,  3680,   473,  3752,    67,
    3824,  -559,  -559,    28,  3896,   538,   554,  3392,  -559,   469,
    3968,   562,   548,   290,   565,   314,  -559,  4040,  5445,   226,
     158,  -559,  -559,  -559,  5805,  5300,  -559,  5805,  3464,  5805,
    5805,  5805,  3536,  5805,  5805,  5805,  5805,  5805,  5805,   306,
    5805,  5805,   403,  -559,  -559,  -559,  -559,  -559,  -559,  -559,
    -559,  3320,  -559,  -559,  -559,  3320,  -559,  -559,  -559,  -559,
    -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,
      92,  3320,    74,  -559,  -559,  -559,  -559,  -559,  -559,  -559,
    -559,  -559,  -559,  -559,  -559,  -559,  -559,  4842,  -559,  -559,
    -559,  -559,  -559,  -559,  -559,  -559,  -559,   139,  -559,    54,
    5805,  -559,   176,  -559,     8,   141,   223,   138,  -559,  4667,
     267,  -559,   286,  -559,   302,   270,  4740,   422,   148,    42,
     445,  4893,   451,   457,  4944,   468,  -559,  3320,   477,  4995,
     501,  -559,   522,  -559,   534,  -559,  5046,   591,   549,   550,
     553,   556,  6348,   559,   566,   218,   596,  -559,  -559,    75,
     602,   125,    50,   131,   608,   397,   124,  -559,   609,  -559,
      82,    82,   616,  5097,  -559,  6348,   569,   619,  -559,  3320,
    -559,  6040,  5517,  -559,   515,  5865,    19,    64,   130,  6593,
     620,  -559,  6348,   126,   393,   393,   220,   621,  -559,   137,
     220,   220,   220,   220,   220,   220,  -559,  -559,  -559,   495,
     220,   220,  -559,  -559,  -559,   627,   631,   154,  -559,  -559,
    -559,  -559,  -559,  -559,   372,  -559,  -559,  -559,  -559,   630,
     199,  -559,  5589,  5373,   628,  5805,  5805,  5805,  5805,  5805,
    5805,  5805,  5805,  5805,  5805,  5805,  5805,  4112,  5805,  5805,
    5805,  5805,  5805,  5805,  5805,  5805,   632,  5805,  5805,   633,
     633,   633,   633,   633,   633,   633,   633,   633,   633,   633,
    -559,  -559,  -559,  5805,  5805,   634,  -559,  -559,  -559,  -559,
    -559,  -559,   635,  -559,   637,  6348,  -559,  -559,   636,  5805,
    -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,
    -559,  -559,  5805,   636,  4184,  -559,  -559,  -559,  -559,  -559,
    -559,  -559,  -559,  -559,  -559,   279,  -559,   356,  -559,  -559,
    -559,  -559,   187,    93,  -559,  -559,  -559,  -559,  -559,  -559,
      59,  -559,  -559,   638,  -559,   640,    90,   156,   645,  -559,
     361,   643,  -559,    84,  -559,   646,  -559,   651,   649,    92,
      92,  -559,  -559,  -559,  -559,  -559,  5805,  -559,  -559,  -559,
     653,  -559,  -559,  6091,  -559,  5661,  5805,  -559,  -559,   584,
    -559,  5805,   595,  -559,   134,  -559,    92,  -559,  -559,  -559,
    -559,  -559,  -559,  -559,  1825,  1480,   499,   607,  1595,   363,
    -559,  -559,  1940,  -559,  3320,  -559,  -559,   140,  -559,   142,
    5805,  5927,  -559,  6348,  6348,  6348,  6348,  6348,  6348,  6348,
    6348,  6348,  6348,  6348,  6397,  -559,  4594,  6544,  6593,   248,
     248,   248,   248,   248,   248,  -559,   393,   393,  -559,  5805,
    5805,  5805,  5805,  5805,  5805,  5805,  5805,  5805,  5805,  5805,
    4791,  6446,  -559,  -559,  -559,   588,  6348,  -559,  6142,  -559,
     661,  6348,   664,   649,  -559,   639,   665,   663,   667,  -559,
    -559,   592,  -559,   669,   670,  -559,  -559,  -559,   674,  -559,
     679,  -559,    27,    48,  -559,  -559,  -559,  -559,  -559,  -559,
     143,  -559,  -559,  6348,  2055,  -559,  -559,  -559,  5989,  6348,
    -559,  6195,  -559,  -559,  -559,   649,  -559,   693,  -559,   532,
    4256,   688,  -559,  -559,  -559,   695,  -559,    73,   415,  -559,
     690,  -559,   697,  -559,   146,   692,  -559,    81,   694,   673,
    -559,  -559,  -559,  -559,   701,  2170,  -559,   650,  -559,   369,
    -559,  -559,  6246,  -559,  5805,  -559,  4328,   227,   227,   487,
     427,   427,   364,   364,   364,   205,   220,   220,  -559,  5805,
    5805,   400,  4400,  -559,   400,  -559,   212,   402,   703,  -559,
     652,   629,  -559,  -559,   558,  -559,  -559,  -559,  -559,   706,
     707,  -559,   708,   710,  -559,   711,   712,  -559,   709,  2285,
    2400,  5805,   292,  5805,  -559,  5805,  -559,  2515,  -559,   716,
    -559,   717,  5148,   718,  -559,  -559,  -559,   407,  -559,  -559,
    -559,   647,   216,  -559,  -559,   720,   414,  -559,   460,  -559,
    -559,   219,  -559,   723,   724,  -559,  -559,  -559,   636,    31,
    -559,   725,  -559,  1710,  -559,   726,   696,   658,  -559,  -559,
     678,  -559,   659,  -559,  6495,   190,  6348,   905,  3320,  -559,
    -559,  4532,  -559,  -559,  -559,  -559,   -22,   733,   735,   734,
     736,   278,  -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,
    5733,  -559,   663,  -559,   738,  -559,  -559,  -559,  -559,  -559,
    -559,  -559,   740,   741,   742,   743,  -559,  -559,  -559,   744,
    6348,  -559,   182,   745,  -559,  -559,  6297,  6348,  -559,  -559,
    -559,  -559,  -559,  2630,  1480,  -559,  -559,    18,  -559,  -559,
      94,  -559,  -559,  3320,  -559,  -559,  -559,  -559,  -559,   576,
    -559,  -559,   746,  -559,   600,   636,  -559,  -559,  -559,   747,
    -559,  -559,   438,   439,   535,  -559,   748,   905,  -559,  -559,
    -559,  -559,  -559,  4472,  -559,   698,  5805,  -559,  -559,   676,
     749,   751,   272,  -559,    17,  -559,  -559,  -559,  -559,  -559,
    -559,  -559,  -559,   109,  -559,  -559,  -559,  -559,  -559,  -559,
    3320,  -559,  -559,  3320,  -559,  2745,  -559,  -559,  3320,  -559,
    3320,  -559,  -559,  -559,   752,  -559,   753,  -559,  3320,   755,
    -559,  3320,   756,  -559,  3320,   757,  -559,  -559,  6348,  -559,
    5199,    92,   109,  -559,  -559,  -559,   759,  -559,  -559,  -559,
     196,  1020,  -559,  1135,  -559,  1250,  -559,  1365,  -559,  -559,
    -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,
    -559,  -559,   761,  -559,  -559,  2860,  2975,  3090,  3205,  -559,
     762,   763,   764,   766,  -559,  -559,  -559,  -559
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -559,  -559,  -559,  -559,  -559,  -306,  -559,    -2,  -559,  -559,
    -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,
    -559,  -559,    66,  -559,  -559,  -559,  -559,  -559,   -47,  -559,
    -559,  -559,  -559,  -559,   208,  -559,  -559,    36,  -559,  -559,
    -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,
    -559,  -559,  -559,  -559,  -559,  -559,  -559,   378,  -559,  -559,
    -559,  -559,    69,  -559,  -559,  -559,  -559,  -559,  -559,  -559,
    -559,  -559,  -559,    62,  -559,  -559,  -559,  -559,  -559,   253,
    -559,  -559,    61,  -559,  -558,  -559,  -559,  -559,  -559,  -559,
    -235,   282,  -332,  -559,  -559,  -559,  -559,  -559,  -559,  -559,
    -559,  -559,  -275,  -559,  -559,  -559,   428,  -559,  -559,  -559,
    -559,  -559,   323,  -559,   132,  -559,  -559,  -559,   232,  -559,
     233,  -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,    -1,
    -559,  -559,  -559,  -559,  -559,  -559,  -559,  -559,   339,  -559,
    -559,  -337,   -10,  -559,   416,   -12,   -37,   781,  -559,  -559,
    -559,  -559,  -559,   641,  -559,  -559,  -559,  -559,  -559,  -559,
    -559,   -35,  -559,  -559,  -559,  -559
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -462
static const yytype_int16 yytable[] =
{
      66,   129,   125,   474,   136,   407,   141,   138,   144,   662,
     196,   297,   149,   203,   122,   156,   675,   209,   162,   123,
     377,   119,   286,     3,   118,   183,   185,   491,   492,   145,
     581,   146,   191,   195,   723,   199,   202,   204,   205,   206,
     202,   210,   211,   212,   213,   214,   215,   487,   220,   221,
     121,   584,   223,   746,   506,   285,   346,   347,   286,   287,
     473,   288,   289,   286,   287,   379,   288,   289,   142,   228,
     282,   147,   809,   230,   607,   239,   378,   286,   342,   608,
     609,   610,   624,   357,   625,   626,   298,   627,   286,   238,
     485,   486,   282,   477,   472,   284,  -322,   284,   286,   282,
     608,   609,   610,   582,   282,   290,   291,   282,   295,   724,
     290,   291,   282,   286,   287,   292,   288,   289,   583,   282,
     313,   380,   725,   490,   585,   282,   345,   354,  -255,   388,
     240,   382,   348,   314,   349,   503,   383,   234,   358,   586,
     390,   538,   381,   540,   587,   321,   282,   618,   282,  -182,
     619,   292,   620,   343,   282,   178,   292,  -255,   282,   479,
     290,   291,   282,   350,   190,   282,   478,   282,   282,   282,
     292,  -322,   628,   282,   282,   282,   282,   282,   282,   296,
     373,   292,   395,   282,   282,   762,   398,   369,   402,  -182,
     470,   292,  -216,   731,   807,   539,   504,   541,   588,   762,
     406,   120,   355,  -255,   284,   383,   292,   409,  -216,   351,
    -432,   611,   505,   653,   283,   284,   299,   284,   505,   708,
     284,   505,   717,   312,  -182,   119,   300,   187,   566,   188,
     202,   411,   480,   413,   414,   415,   416,   417,   418,   419,
     420,   421,   422,   423,   424,   426,   427,   428,   429,   430,
     431,   432,   433,   434,  -216,   436,   437,   763,   282,   709,
     764,   242,   718,   243,   244,   471,   301,   654,   284,   189,
     304,   450,   451,   307,   764,   805,   242,  -216,   243,   244,
     462,   806,  -268,   242,   750,   243,   244,   456,   455,   305,
     505,   170,   663,   340,   710,   691,   171,   719,   692,   676,
     458,   693,   461,   459,   242,   306,   243,   244,   278,   279,
     216,  -268,   217,   308,   751,   178,   659,   694,   280,   281,
     179,   494,   172,   271,   272,   273,   274,   275,   276,   277,
     278,   279,   218,   280,   281,   463,   282,   266,   267,   268,
     280,   281,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   130,   493,   131,   535,   466,   132,  -272,
     133,   280,   281,   498,   499,   219,   532,   482,   483,   501,
    -432,   134,   638,   399,   282,   400,   282,   282,   282,   282,
     282,   282,   282,   282,   282,   282,   282,   282,   467,   282,
     282,   282,   282,   282,   282,   282,   282,   282,   542,   282,
     282,   771,   536,   647,   611,   655,   533,   222,   656,   123,
     705,   657,   639,   282,   282,   401,   614,   712,  -153,   282,
     242,   282,   243,   244,   282,   311,   810,   547,   548,   549,
     550,   551,   552,   553,   554,   555,   556,   557,   658,   786,
     789,   787,   790,   648,   589,   590,   659,   660,   315,   242,
     706,   243,   244,   119,   317,   661,   282,   713,  -153,   597,
     318,   282,   282,   715,   282,   832,   277,   278,   279,   831,
     158,   320,   353,   120,   137,   159,   160,   280,   281,   123,
     322,   788,   791,   242,   633,   243,   244,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   602,   391,
     515,   392,   516,   716,   324,   282,   280,   281,  -149,   642,
     282,   282,   282,   282,   282,   282,   282,   282,   282,   282,
     282,   393,   517,   518,   645,   326,   274,   275,   276,   277,
     278,   279,   202,   599,   644,   600,   792,   328,   793,   150,
     280,   281,  -152,   242,   151,   243,   244,   202,   646,   166,
     651,   167,   334,   335,   168,   152,   336,   703,   704,   337,
     153,   673,   338,   164,   656,   282,   174,   674,   165,   339,
     363,   175,   374,  -461,  -461,  -461,  -461,  -461,   794,   690,
     619,   696,   620,   697,   272,   273,   274,   275,   276,   277,
     278,   279,   330,   575,   658,  -461,  -461,   331,   331,   341,
     280,   281,   659,   660,   626,   344,   627,   282,   522,   282,
     523,   352,   356,  -461,   282,  -461,  -149,  -461,   722,   361,
    -461,  -461,   367,   387,   389,  -461,   364,  -461,   150,  -461,
     524,   518,   152,   405,   412,   754,   438,   452,   435,   453,
     454,   500,   123,   476,   475,   735,   741,   365,   481,   484,
    -152,  -461,   175,   282,   489,   383,   495,   502,   202,   282,
     282,  -461,  -461,   560,   563,   775,  -461,   565,   569,   570,
     573,   467,   577,   578,  -461,  -461,  -461,  -461,  -461,  -461,
     579,  -461,  -461,  -461,  -461,   580,   440,   441,   442,   443,
     444,   445,   446,   447,   448,   449,   598,   603,   606,   615,
     617,   623,   527,   631,   634,   636,   669,   672,   670,   680,
     681,   776,   686,   729,   682,   784,   683,   684,   685,   699,
     700,   702,   811,   711,   707,   813,   720,   721,   726,   727,
     815,   798,   817,   730,   800,   735,   747,   284,   748,   728,
     179,   756,   749,   757,   758,   759,   760,   761,   766,   782,
     785,   802,   803,   799,   804,   819,   820,   795,   823,   826,
     829,   282,   833,   282,   839,   844,   845,   846,   812,   847,
     768,   814,   652,   796,   835,   525,   816,   836,   818,   774,
     837,   781,   632,   488,   838,   783,   822,   596,   568,   825,
      -2,     4,   828,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,   755,    16,   677,   678,    17,   834,
     576,   163,    18,    19,     0,    20,    21,    22,    23,     0,
      24,    25,   360,    26,    27,    28,     0,    29,    30,    31,
      32,    33,    34,     0,    35,     0,    36,    37,    38,    39,
      40,    41,    42,     0,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,  -106,    12,    13,    14,    15,     0,
      16,     0,     0,    17,   732,   733,   734,    18,     0,     0,
      20,    21,    22,    23,     0,    24,   225,     0,   226,    27,
      28,     0,     0,    30,     0,     0,     0,     0,     0,   227,
       0,    36,    37,    38,    39,     0,    41,    42,     0,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,  -146,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,  -146,  -146,    20,    21,    22,    23,     0,
      24,   225,     0,   226,    27,    28,     0,     0,    30,     0,
       0,     0,     0,  -146,   227,     0,    36,    37,    38,    39,
       0,    41,    42,     0,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,  -142,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,  -142,  -142,
      20,    21,    22,    23,     0,    24,   225,     0,   226,    27,
      28,     0,     0,    30,     0,     0,     0,     0,  -142,   227,
       0,    36,    37,    38,    39,     0,    41,    42,     0,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,  -177,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,  -177,  -177,    20,    21,    22,    23,     0,
      24,   225,     0,   226,    27,    28,     0,     0,    30,     0,
       0,     0,     0,  -177,   227,     0,    36,    37,    38,    39,
       0,    41,    42,     0,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,  -173,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,  -173,  -173,
      20,    21,    22,    23,     0,    24,   225,     0,   226,    27,
      28,     0,     0,    30,     0,     0,     0,     0,  -173,   227,
       0,    36,    37,    38,    39,     0,    41,    42,     0,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,   -75,
      12,    13,    14,    15,     0,    16,   509,   510,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   225,     0,   226,    27,    28,     0,     0,    30,     0,
       0,     0,     0,     0,   227,     0,    36,    37,    38,    39,
       0,    41,    42,     0,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,  -190,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,   527,    24,   225,     0,   226,    27,
      28,     0,     0,    30,     0,     0,     0,     0,     0,   227,
       0,    36,    37,    38,    39,     0,    41,    42,     0,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,  -194,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,  -194,
      24,   225,     0,   226,    27,    28,     0,     0,    30,     0,
       0,     0,     0,     0,   227,     0,    36,    37,    38,    39,
       0,    41,    42,     0,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,   507,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,     0,    24,   225,     0,   226,    27,
      28,     0,     0,    30,     0,     0,     0,     0,     0,   227,
       0,    36,    37,    38,    39,     0,    41,    42,     0,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,   534,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   225,     0,   226,    27,    28,     0,     0,    30,     0,
       0,     0,     0,     0,   227,     0,    36,    37,    38,    39,
       0,    41,    42,     0,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,   591,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,     0,    24,   225,     0,   226,    27,
      28,     0,     0,    30,     0,     0,     0,     0,     0,   227,
       0,    36,    37,    38,    39,     0,    41,    42,     0,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,   635,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   225,     0,   226,    27,    28,     0,     0,    30,     0,
       0,     0,     0,     0,   227,     0,    36,    37,    38,    39,
       0,    41,    42,     0,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,   687,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,     0,    24,   225,     0,   226,    27,
      28,     0,     0,    30,     0,     0,     0,     0,     0,   227,
       0,    36,    37,    38,    39,     0,    41,    42,     0,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,   688,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   225,     0,   226,    27,    28,     0,     0,    30,     0,
       0,     0,     0,     0,   227,     0,    36,    37,    38,    39,
       0,    41,    42,     0,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,     0,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,     0,    24,   225,     0,   226,    27,
      28,     0,     0,    30,     0,     0,     0,     0,     0,   227,
       0,    36,    37,    38,    39,     0,    41,    42,     0,    43,
       0,    44,     0,    45,     0,    46,   698,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,   -78,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   225,     0,   226,    27,    28,     0,     0,    30,     0,
       0,     0,     0,     0,   227,     0,    36,    37,    38,    39,
       0,    41,    42,     0,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,  -155,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,     0,    24,   225,     0,   226,    27,
      28,     0,     0,    30,     0,     0,     0,     0,     0,   227,
       0,    36,    37,    38,    39,     0,    41,    42,     0,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,   840,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   225,     0,   226,    27,    28,     0,     0,    30,     0,
       0,     0,     0,     0,   227,     0,    36,    37,    38,    39,
       0,    41,    42,     0,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,   841,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,     0,    24,   225,     0,   226,    27,
      28,     0,     0,    30,     0,     0,     0,     0,     0,   227,
       0,    36,    37,    38,    39,     0,    41,    42,     0,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,   842,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   225,     0,   226,    27,    28,     0,     0,    30,     0,
       0,     0,     0,     0,   227,     0,    36,    37,    38,    39,
       0,    41,    42,     0,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,   843,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,     0,    24,   225,     0,   226,    27,
      28,     0,     0,    30,     0,     0,     0,     0,     0,   227,
       0,    36,    37,    38,    39,     0,    41,    42,     0,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,     0,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   225,     0,   226,    27,    28,     0,     0,    30,     0,
       0,     0,     0,     0,   227,     0,    36,    37,    38,    39,
       0,    41,    42,     0,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   154,     0,   155,     6,     7,   127,     9,
      10,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,    21,    22,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,   128,     0,    36,     0,
      38,     0,     0,    41,    42,     0,     0,     0,    44,     0,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   200,     0,   201,     6,     7,
     127,     9,    10,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
      21,    22,     0,     0,     0,     0,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,   128,     0,
      36,     0,    38,     0,     0,    41,    42,     0,     0,     0,
      44,     0,    45,     0,    46,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   207,     0,   208,
       6,     7,   127,     9,    10,     0,    47,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    49,    50,     0,     0,
       0,    51,    21,    22,     0,     0,     0,     0,     0,    53,
      54,    55,    56,    57,    58,     0,    59,    60,    61,    62,
     128,     0,    36,     0,    38,     0,     0,    41,    42,     0,
       0,     0,    44,     0,    45,     0,    46,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   126,
       0,     0,     6,     7,   127,     9,    10,     0,    47,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    49,    50,
       0,     0,     0,    51,    21,    22,     0,     0,     0,     0,
       0,    53,    54,    55,    56,    57,    58,     0,    59,    60,
      61,    62,   128,     0,    36,     0,    38,     0,     0,    41,
      42,     0,     0,     0,    44,     0,    45,     0,    46,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   135,     0,     0,     6,     7,   127,     9,    10,     0,
      47,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      49,    50,     0,     0,     0,    51,    21,    22,     0,     0,
       0,     0,     0,    53,    54,    55,    56,    57,    58,     0,
      59,    60,    61,    62,   128,     0,    36,     0,    38,     0,
       0,    41,    42,     0,     0,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   140,     0,     0,     6,     7,   127,     9,
      10,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,    21,    22,
       0,     0,     0,     0,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,   128,     0,    36,     0,
      38,     0,     0,    41,    42,     0,     0,     0,    44,     0,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   143,     0,     0,     6,     7,
     127,     9,    10,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
      21,    22,     0,     0,     0,     0,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,   128,     0,
      36,     0,    38,     0,     0,    41,    42,     0,     0,     0,
      44,     0,    45,     0,    46,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   148,     0,     0,
       6,     7,   127,     9,    10,     0,    47,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    49,    50,     0,     0,
       0,    51,    21,    22,     0,     0,     0,     0,     0,    53,
      54,    55,    56,    57,    58,     0,    59,    60,    61,    62,
     128,     0,    36,     0,    38,     0,     0,    41,    42,     0,
       0,     0,    44,     0,    45,     0,    46,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   161,
       0,     0,     6,     7,   127,     9,    10,     0,    47,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    49,    50,
       0,     0,     0,    51,    21,    22,     0,     0,     0,     0,
       0,    53,    54,    55,    56,    57,    58,     0,    59,    60,
      61,    62,   128,     0,    36,     0,    38,     0,     0,    41,
      42,     0,     0,     0,    44,     0,    45,     0,    46,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   182,     0,     0,     6,     7,   127,     9,    10,     0,
      47,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      49,    50,     0,     0,     0,    51,    21,    22,     0,     0,
       0,     0,     0,    53,    54,    55,    56,    57,    58,     0,
      59,    60,    61,    62,   128,     0,    36,     0,    38,     0,
       0,    41,    42,     0,     0,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   425,     0,     0,     6,     7,   127,     9,
      10,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,    21,    22,
       0,     0,     0,     0,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,   128,     0,    36,     0,
      38,     0,     0,    41,    42,     0,     0,     0,    44,     0,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   460,     0,     0,     6,     7,
     127,     9,    10,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
      21,    22,     0,     0,     0,     0,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,   128,     0,
      36,     0,    38,     0,     0,    41,    42,     0,     0,     0,
      44,     0,    45,     0,    46,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   601,     0,     0,
       6,     7,   127,     9,    10,     0,    47,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    49,    50,     0,     0,
       0,    51,    21,    22,     0,     0,     0,     0,     0,    53,
      54,    55,    56,    57,    58,     0,    59,    60,    61,    62,
     128,     0,    36,     0,    38,     0,     0,    41,    42,     0,
       0,     0,    44,     0,    45,     0,    46,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   643,
       0,     0,     6,     7,   127,     9,    10,     0,    47,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    49,    50,
       0,     0,     0,    51,    21,    22,     0,     0,     0,     0,
       0,    53,    54,    55,    56,    57,    58,     0,    59,    60,
      61,    62,   128,     0,    36,     0,    38,     0,     0,    41,
      42,     0,     0,     0,    44,     0,    45,     0,    46,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   650,     0,     0,     6,     7,   127,     9,    10,     0,
      47,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      49,    50,     0,     0,     0,    51,    21,    22,     0,     0,
       0,     0,     0,    53,    54,    55,    56,    57,    58,     0,
      59,    60,    61,    62,   128,     0,    36,     0,    38,     0,
       0,    41,    42,     0,     0,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   797,     0,     0,     6,     7,   127,     9,
      10,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,    21,    22,
       0,     0,     0,     0,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,   128,     0,    36,     0,
      38,     0,     0,    41,    42,     0,     0,     0,    44,     0,
      45,     0,    46,   742,     0,  -103,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,  -103,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,   242,     0,
     243,   244,     0,     0,     0,   545,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,     0,     0,
     743,   257,   258,   259,     0,   260,   261,   262,   263,   264,
     265,   266,   267,   268,     0,     0,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   546,     0,     0,
       0,     0,     0,     0,     0,   280,   281,     0,     0,     0,
     242,     0,   243,   244,     0,     0,     0,     0,   245,   246,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
     302,     0,     0,   257,   258,   259,     0,   260,   261,   262,
     263,   264,   265,   266,   267,   268,     0,     0,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,     0,
       0,     0,     0,     0,     0,     0,     0,   280,   281,     0,
     303,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   242,     0,   243,   244,     0,     0,     0,
       0,   245,   246,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,   309,     0,     0,   257,   258,   259,     0,
     260,   261,   262,   263,   264,   265,   266,   267,   268,     0,
       0,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,     0,     0,     0,     0,     0,     0,     0,     0,
     280,   281,     0,   310,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   558,     0,   242,     0,   243,   244,
       0,     0,     0,     0,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,     0,     0,     0,   257,
     258,   259,     0,   260,   261,   262,   263,   264,   265,   266,
     267,   268,     0,     0,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   241,     0,   242,     0,   243,
     244,     0,     0,   280,   281,   245,   246,   247,   248,   249,
     250,   251,   252,   253,   254,   255,   256,     0,     0,   559,
     257,   258,   259,     0,   260,   261,   262,   263,   264,   265,
     266,   267,   268,     0,     0,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   316,     0,   242,     0,
     243,   244,     0,     0,   280,   281,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,     0,     0,
       0,   257,   258,   259,     0,   260,   261,   262,   263,   264,
     265,   266,   267,   268,     0,     0,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   319,     0,   242,
       0,   243,   244,     0,     0,   280,   281,   245,   246,   247,
     248,   249,   250,   251,   252,   253,   254,   255,   256,     0,
       0,     0,   257,   258,   259,     0,   260,   261,   262,   263,
     264,   265,   266,   267,   268,     0,     0,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   323,     0,
     242,     0,   243,   244,     0,     0,   280,   281,   245,   246,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
       0,     0,     0,   257,   258,   259,     0,   260,   261,   262,
     263,   264,   265,   266,   267,   268,     0,     0,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   329,
       0,   242,     0,   243,   244,     0,     0,   280,   281,   245,
     246,   247,   248,   249,   250,   251,   252,   253,   254,   255,
     256,     0,     0,     0,   257,   258,   259,     0,   260,   261,
     262,   263,   264,   265,   266,   267,   268,     0,     0,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   279,
     362,     0,   242,     0,   243,   244,     0,     0,   280,   281,
     245,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,     0,     0,     0,   257,   258,   259,     0,   260,
     261,   262,   263,   264,   265,   266,   267,   268,     0,     0,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,   701,     0,   242,     0,   243,   244,     0,     0,   280,
     281,   245,   246,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,     0,     0,     0,   257,   258,   259,     0,
     260,   261,   262,   263,   264,   265,   266,   267,   268,     0,
       0,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,   830,     0,   242,     0,   243,   244,     0,     0,
     280,   281,   245,   246,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,     0,     0,     0,   257,   258,   259,
       0,   260,   261,   262,   263,   264,   265,   266,   267,   268,
       0,     0,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,     0,     0,   242,     0,   243,   244,     0,
       0,   280,   281,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,     0,     0,     0,   257,   258,
     259,     0,   260,   261,   262,   263,   264,   265,   266,   267,
     268,     0,     0,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,     6,     7,   127,     9,    10,     0,
       0,     0,   280,   281,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    21,    22,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   192,   128,     0,    36,     0,    38,     0,
       0,    41,    42,     0,     0,     0,    44,   193,    45,     0,
      46,     0,   194,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     6,     7,   127,
       9,    10,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,    21,
      22,     0,     0,     0,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,   192,   128,     0,    36,
       0,    38,     0,     0,    41,    42,     0,     0,     0,    44,
       0,    45,     0,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     6,
       7,   127,     9,    10,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,    50,     0,     0,     0,
      51,    21,    22,     0,   410,     0,     0,     0,    53,    54,
      55,    56,    57,    58,     0,    59,    60,    61,    62,   128,
       0,    36,     0,    38,     0,     0,    41,    42,     0,     0,
       0,    44,   184,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     6,     7,   127,     9,    10,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,    21,    22,     0,     0,     0,     0,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,   128,     0,    36,     0,    38,     0,     0,    41,    42,
       0,     0,     0,    44,   372,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     6,     7,   127,     9,    10,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,    21,    22,     0,     0,     0,
       0,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,   128,     0,    36,     0,    38,     0,     0,
      41,    42,     0,     0,   408,    44,     0,    45,     0,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     6,     7,   127,     9,    10,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,    50,     0,     0,     0,    51,    21,    22,     0,
       0,     0,     0,     0,    53,    54,    55,    56,    57,    58,
       0,    59,    60,    61,    62,   128,     0,    36,     0,    38,
       0,     0,    41,    42,     0,     0,     0,    44,   497,    45,
       0,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     6,     7,   127,
       9,    10,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    49,    50,     0,     0,     0,    51,    21,
      22,     0,     0,     0,     0,     0,    53,    54,    55,    56,
      57,    58,     0,    59,    60,    61,    62,   128,     0,    36,
       0,    38,     0,     0,    41,    42,     0,     0,   753,    44,
       0,    45,     0,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     6,
       7,   127,     9,    10,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,    50,     0,     0,     0,
      51,    21,    22,     0,     0,     0,     0,     0,    53,    54,
      55,    56,    57,    58,     0,    59,    60,    61,    62,   128,
       0,    36,     0,    38,     0,     0,    41,    42,     0,     0,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,   375,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,   242,     0,   243,   244,     0,     0,   376,     0,   245,
     246,   247,   248,   249,   250,   251,   252,   253,   254,   255,
     256,     0,     0,     0,   257,   258,   259,     0,   260,   261,
     262,   263,   264,   265,   266,   267,   268,     0,     0,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   279,
     375,     0,     0,     0,     0,     0,     0,     0,   280,   281,
       0,     0,     0,   242,   543,   243,   244,     0,     0,     0,
       0,   245,   246,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,     0,     0,     0,   257,   258,   259,     0,
     260,   261,   262,   263,   264,   265,   266,   267,   268,     0,
       0,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,   593,     0,     0,     0,     0,     0,     0,     0,
     280,   281,     0,     0,     0,   242,   594,   243,   244,     0,
       0,     0,     0,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,     0,     0,     0,   257,   258,
     259,     0,   260,   261,   262,   263,   264,   265,   266,   267,
     268,     0,     0,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,     0,   371,   242,     0,   243,   244,
       0,     0,   280,   281,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,     0,     0,     0,   257,
     258,   259,     0,   260,   261,   262,   263,   264,   265,   266,
     267,   268,     0,     0,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,     0,     0,   242,   496,   243,
     244,     0,     0,   280,   281,   245,   246,   247,   248,   249,
     250,   251,   252,   253,   254,   255,   256,     0,     0,     0,
     257,   258,   259,     0,   260,   261,   262,   263,   264,   265,
     266,   267,   268,     0,     0,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,     0,     0,   242,     0,
     243,   244,     0,     0,   280,   281,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,     0,   562,
       0,   257,   258,   259,     0,   260,   261,   262,   263,   264,
     265,   266,   267,   268,     0,     0,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,     0,     0,     0,
       0,   242,     0,   243,   244,   280,   281,   595,     0,   245,
     246,   247,   248,   249,   250,   251,   252,   253,   254,   255,
     256,     0,     0,     0,   257,   258,   259,     0,   260,   261,
     262,   263,   264,   265,   266,   267,   268,     0,     0,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   279,
       0,     0,   242,   641,   243,   244,     0,     0,   280,   281,
     245,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,     0,     0,     0,   257,   258,   259,     0,   260,
     261,   262,   263,   264,   265,   266,   267,   268,     0,     0,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,     0,     0,   242,   767,   243,   244,     0,     0,   280,
     281,   245,   246,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,     0,     0,     0,   257,   258,   259,     0,
     260,   261,   262,   263,   264,   265,   266,   267,   268,     0,
       0,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,     0,     0,   242,     0,   243,   244,     0,     0,
     280,   281,   245,   246,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,     0,     0,     0,   257,   258,   259,
       0,   260,   261,   262,   263,   264,   265,   266,   267,   268,
       0,     0,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   242,     0,   243,   244,     0,     0,     0,
       0,   280,   281,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   256,     0,     0,   544,   257,   258,   259,     0,
     260,   261,   262,   263,   264,   265,   266,   267,   268,     0,
       0,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,   242,     0,   243,   244,     0,     0,     0,     0,
     280,   281,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   257,   258,   259,     0,   260,
     261,   262,   263,   264,   265,   266,   267,   268,     0,     0,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,   242,     0,   243,   244,     0,     0,     0,     0,   280,
     281,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   258,   259,     0,   260,   261,
     262,   263,   264,   265,   266,   267,   268,     0,     0,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   279,
     242,     0,   243,   244,     0,     0,     0,     0,   280,   281,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   259,     0,   260,   261,   262,
     263,   264,   265,   266,   267,   268,     0,     0,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   242,
       0,   243,   244,     0,     0,     0,     0,   280,   281,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   260,   261,   262,   263,
     264,   265,   266,   267,   268,     0,     0,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,     0,     0,
       0,     0,     0,     0,     0,     0,   280,   281
};

static const yytype_int16 yycheck[] =
{
       2,    13,    12,   340,    16,   240,    18,    17,    20,   567,
      45,     3,    24,    48,     1,    27,   574,    52,    30,     6,
       1,    43,     4,     0,     3,    37,    38,   359,   360,     1,
       3,     3,    44,    45,     3,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,   353,    60,    61,
       3,     3,    62,    75,   386,     1,     6,     7,     4,     5,
       1,     7,     8,     4,     5,     1,     7,     8,     1,    71,
     107,    43,    55,    75,     1,     1,    57,     4,     3,     6,
       7,     8,     1,     1,     3,     4,    78,     6,     4,    91,
       6,     7,   129,     3,     1,    78,     3,    78,     4,   136,
       6,     7,     8,    76,   141,    51,    52,   144,   120,    78,
      51,    52,   149,     4,     5,    97,     7,     8,    91,   156,
      78,    57,    91,   358,    76,   162,     1,     3,     3,     3,
      56,     1,     1,    91,     3,     1,     6,    45,    56,    91,
       3,     1,    78,     1,     1,   147,   183,     1,   185,     3,
       4,    97,     6,    78,   191,     1,    97,    32,   195,     3,
      51,    52,   199,    32,     6,   202,    76,   204,   205,   206,
      97,    78,    91,   210,   211,   212,   213,   214,   215,     3,
     192,    97,   229,   220,   221,     3,   233,   189,   235,    43,
       3,    97,    62,     3,   752,    55,    62,    55,    55,     3,
       1,    63,    78,    78,    78,     6,    97,   242,    78,    78,
      56,   517,    78,     1,    75,    78,    75,    78,    78,     3,
      78,    78,     3,    75,    78,    43,     3,     1,   463,     3,
     242,   243,    76,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,    55,   267,   268,    75,   295,    43,
      78,    56,    43,    58,    59,    78,    43,    55,    78,    43,
       3,   283,   284,     3,    78,     3,    56,    78,    58,    59,
       1,     9,     3,    56,     6,    58,    59,   299,   298,     3,
      78,     1,   567,    75,    78,     3,     6,    78,     6,   574,
     312,     9,   314,   313,    56,     3,    58,    59,   103,   104,
       4,    32,     6,    43,    36,     1,    44,   592,   113,   114,
       6,   368,    32,    96,    97,    98,    99,   100,   101,   102,
     103,   104,    26,   113,   114,    56,   373,    89,    90,    91,
     113,   114,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,     1,   366,     3,   403,     1,     1,     3,
       3,   113,   114,   375,   376,    59,     3,     6,     7,   381,
      56,    14,     3,     1,   411,     3,   413,   414,   415,   416,
     417,   418,   419,   420,   421,   422,   423,   424,    32,   426,
     427,   428,   429,   430,   431,   432,   433,   434,   410,   436,
     437,   707,   404,     3,   710,     3,    43,     4,     6,     6,
       3,     9,    43,   450,   451,    43,     1,     3,     3,   456,
      56,   458,    58,    59,   461,     3,   763,   439,   440,   441,
     442,   443,   444,   445,   446,   447,   448,   449,    36,     1,
       1,     3,     3,    43,   491,   492,    44,    45,     3,    56,
      43,    58,    59,    43,     3,    53,   493,    43,    43,   506,
       3,   498,   499,     3,   501,   802,   102,   103,   104,   801,
       1,     3,    75,    63,     1,     6,     7,   113,   114,     6,
       3,    43,    43,    56,   531,    58,    59,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   510,     4,
       1,     6,     3,    43,     3,   542,   113,   114,     9,   544,
     547,   548,   549,   550,   551,   552,   553,   554,   555,   556,
     557,    26,    23,    24,   559,     3,    99,   100,   101,   102,
     103,   104,   544,     1,   546,     3,     1,     3,     3,     1,
     113,   114,    43,    56,     6,    58,    59,   559,   560,     1,
     562,     3,     3,     3,     6,     1,     3,   604,   605,     3,
       6,     3,     3,     1,     6,   602,     1,     9,     6,     3,
       1,     6,    57,     4,     5,     6,     7,     8,    43,   591,
       4,   593,     6,   595,    97,    98,    99,   100,   101,   102,
     103,   104,     1,     1,    36,    26,    27,     6,     6,     3,
     113,   114,    44,    45,     4,     3,     6,   644,     1,   646,
       3,     3,     3,    44,   651,    46,     9,    48,   628,     3,
      51,    52,     3,     3,     3,    56,    57,    58,     1,    60,
      23,    24,     1,     3,     6,   670,     3,     3,     6,     4,
       3,    57,     6,     3,     6,   647,   648,    78,     3,     6,
      43,    82,     6,   690,     3,     6,     3,    62,   670,   696,
     697,    92,    93,    75,     3,   712,    97,     3,     3,     6,
       3,    32,     3,     3,   105,   106,   107,   108,   109,   110,
       6,   112,   113,   114,   115,     6,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,     3,     9,     3,     9,
       3,     9,    29,     9,     3,    55,     3,    78,    56,     3,
       3,   713,     3,    55,     6,   725,     6,     6,     6,     3,
       3,     3,   769,     3,    77,   772,     3,     3,     3,     3,
     777,   743,   779,    55,   746,   737,     3,    78,     3,    43,
       6,     3,     6,     3,     3,     3,     3,     3,     3,     3,
       3,    75,     3,    55,     3,     3,     3,     9,     3,     3,
       3,   798,     3,   800,     3,     3,     3,     3,   770,     3,
     704,   773,   564,   737,   821,   397,   778,   824,   780,   710,
     827,   719,   529,   355,   831,   724,   788,   505,   465,   791,
       0,     1,   794,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,   672,    15,   574,   574,    18,   810,
     471,    30,    22,    23,    -1,    25,    26,    27,    28,    -1,
      30,    31,   181,    33,    34,    35,    -1,    37,    38,    39,
      40,    41,    42,    -1,    44,    -1,    46,    47,    48,    49,
      50,    51,    52,    -1,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    19,    20,    21,    22,    -1,    -1,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    38,    -1,    -1,    -1,    -1,    -1,    44,
      -1,    46,    47,    48,    49,    -1,    51,    52,    -1,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    23,    24,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    38,    -1,
      -1,    -1,    -1,    43,    44,    -1,    46,    47,    48,    49,
      -1,    51,    52,    -1,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    38,    -1,    -1,    -1,    -1,    43,    44,
      -1,    46,    47,    48,    49,    -1,    51,    52,    -1,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    23,    24,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    38,    -1,
      -1,    -1,    -1,    43,    44,    -1,    46,    47,    48,    49,
      -1,    51,    52,    -1,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    38,    -1,    -1,    -1,    -1,    43,    44,
      -1,    46,    47,    48,    49,    -1,    51,    52,    -1,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    16,    17,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    38,    -1,
      -1,    -1,    -1,    -1,    44,    -1,    46,    47,    48,    49,
      -1,    51,    52,    -1,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    38,    -1,    -1,    -1,    -1,    -1,    44,
      -1,    46,    47,    48,    49,    -1,    51,    52,    -1,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    38,    -1,
      -1,    -1,    -1,    -1,    44,    -1,    46,    47,    48,    49,
      -1,    51,    52,    -1,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    38,    -1,    -1,    -1,    -1,    -1,    44,
      -1,    46,    47,    48,    49,    -1,    51,    52,    -1,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    38,    -1,
      -1,    -1,    -1,    -1,    44,    -1,    46,    47,    48,    49,
      -1,    51,    52,    -1,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    38,    -1,    -1,    -1,    -1,    -1,    44,
      -1,    46,    47,    48,    49,    -1,    51,    52,    -1,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    38,    -1,
      -1,    -1,    -1,    -1,    44,    -1,    46,    47,    48,    49,
      -1,    51,    52,    -1,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    38,    -1,    -1,    -1,    -1,    -1,    44,
      -1,    46,    47,    48,    49,    -1,    51,    52,    -1,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    38,    -1,
      -1,    -1,    -1,    -1,    44,    -1,    46,    47,    48,    49,
      -1,    51,    52,    -1,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,    -1,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    38,    -1,    -1,    -1,    -1,    -1,    44,
      -1,    46,    47,    48,    49,    -1,    51,    52,    -1,    54,
      -1,    56,    -1,    58,    -1,    60,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    38,    -1,
      -1,    -1,    -1,    -1,    44,    -1,    46,    47,    48,    49,
      -1,    51,    52,    -1,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    38,    -1,    -1,    -1,    -1,    -1,    44,
      -1,    46,    47,    48,    49,    -1,    51,    52,    -1,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    38,    -1,
      -1,    -1,    -1,    -1,    44,    -1,    46,    47,    48,    49,
      -1,    51,    52,    -1,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    38,    -1,    -1,    -1,    -1,    -1,    44,
      -1,    46,    47,    48,    49,    -1,    51,    52,    -1,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    38,    -1,
      -1,    -1,    -1,    -1,    44,    -1,    46,    47,    48,    49,
      -1,    51,    52,    -1,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    38,    -1,    -1,    -1,    -1,    -1,    44,
      -1,    46,    47,    48,    49,    -1,    51,    52,    -1,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,    -1,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    38,    -1,
      -1,    -1,    -1,    -1,    44,    -1,    46,    47,    48,    49,
      -1,    51,    52,    -1,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     1,    -1,     3,     4,     5,     6,     7,
       8,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    26,    27,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,    44,    -1,    46,    -1,
      48,    -1,    -1,    51,    52,    -1,    -1,    -1,    56,    -1,
      58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     1,    -1,     3,     4,     5,
       6,     7,     8,    -1,    82,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      26,    27,    -1,    -1,    -1,    -1,    -1,   105,   106,   107,
     108,   109,   110,    -1,   112,   113,   114,   115,    44,    -1,
      46,    -1,    48,    -1,    -1,    51,    52,    -1,    -1,    -1,
      56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,     3,
       4,     5,     6,     7,     8,    -1,    82,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,
      -1,    97,    26,    27,    -1,    -1,    -1,    -1,    -1,   105,
     106,   107,   108,   109,   110,    -1,   112,   113,   114,   115,
      44,    -1,    46,    -1,    48,    -1,    -1,    51,    52,    -1,
      -1,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    82,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,
      -1,    -1,    -1,    97,    26,    27,    -1,    -1,    -1,    -1,
      -1,   105,   106,   107,   108,   109,   110,    -1,   112,   113,
     114,   115,    44,    -1,    46,    -1,    48,    -1,    -1,    51,
      52,    -1,    -1,    -1,    56,    -1,    58,    -1,    60,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      92,    93,    -1,    -1,    -1,    97,    26,    27,    -1,    -1,
      -1,    -1,    -1,   105,   106,   107,   108,   109,   110,    -1,
     112,   113,   114,   115,    44,    -1,    46,    -1,    48,    -1,
      -1,    51,    52,    -1,    -1,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    26,    27,
      -1,    -1,    -1,    -1,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,    44,    -1,    46,    -1,
      48,    -1,    -1,    51,    52,    -1,    -1,    -1,    56,    -1,
      58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    82,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      26,    27,    -1,    -1,    -1,    -1,    -1,   105,   106,   107,
     108,   109,   110,    -1,   112,   113,   114,   115,    44,    -1,
      46,    -1,    48,    -1,    -1,    51,    52,    -1,    -1,    -1,
      56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    82,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,
      -1,    97,    26,    27,    -1,    -1,    -1,    -1,    -1,   105,
     106,   107,   108,   109,   110,    -1,   112,   113,   114,   115,
      44,    -1,    46,    -1,    48,    -1,    -1,    51,    52,    -1,
      -1,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    82,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,
      -1,    -1,    -1,    97,    26,    27,    -1,    -1,    -1,    -1,
      -1,   105,   106,   107,   108,   109,   110,    -1,   112,   113,
     114,   115,    44,    -1,    46,    -1,    48,    -1,    -1,    51,
      52,    -1,    -1,    -1,    56,    -1,    58,    -1,    60,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      92,    93,    -1,    -1,    -1,    97,    26,    27,    -1,    -1,
      -1,    -1,    -1,   105,   106,   107,   108,   109,   110,    -1,
     112,   113,   114,   115,    44,    -1,    46,    -1,    48,    -1,
      -1,    51,    52,    -1,    -1,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    26,    27,
      -1,    -1,    -1,    -1,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,    44,    -1,    46,    -1,
      48,    -1,    -1,    51,    52,    -1,    -1,    -1,    56,    -1,
      58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    82,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      26,    27,    -1,    -1,    -1,    -1,    -1,   105,   106,   107,
     108,   109,   110,    -1,   112,   113,   114,   115,    44,    -1,
      46,    -1,    48,    -1,    -1,    51,    52,    -1,    -1,    -1,
      56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    82,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,
      -1,    97,    26,    27,    -1,    -1,    -1,    -1,    -1,   105,
     106,   107,   108,   109,   110,    -1,   112,   113,   114,   115,
      44,    -1,    46,    -1,    48,    -1,    -1,    51,    52,    -1,
      -1,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    82,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,
      -1,    -1,    -1,    97,    26,    27,    -1,    -1,    -1,    -1,
      -1,   105,   106,   107,   108,   109,   110,    -1,   112,   113,
     114,   115,    44,    -1,    46,    -1,    48,    -1,    -1,    51,
      52,    -1,    -1,    -1,    56,    -1,    58,    -1,    60,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      92,    93,    -1,    -1,    -1,    97,    26,    27,    -1,    -1,
      -1,    -1,    -1,   105,   106,   107,   108,   109,   110,    -1,
     112,   113,   114,   115,    44,    -1,    46,    -1,    48,    -1,
      -1,    51,    52,    -1,    -1,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    26,    27,
      -1,    -1,    -1,    -1,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,    44,    -1,    46,    -1,
      48,    -1,    -1,    51,    52,    -1,    -1,    -1,    56,    -1,
      58,    -1,    60,     1,    -1,     3,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      -1,    -1,    -1,    -1,    -1,    43,    -1,   105,   106,   107,
     108,   109,   110,    -1,   112,   113,   114,   115,    56,    -1,
      58,    59,    -1,    -1,    -1,     1,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      78,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    -1,    -1,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,    43,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   113,   114,    -1,    -1,    -1,
      56,    -1,    58,    59,    -1,    -1,    -1,    -1,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
       3,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    -1,    -1,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   113,   114,    -1,
      43,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    56,    -1,    58,    59,    -1,    -1,    -1,
      -1,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,     3,    -1,    -1,    79,    80,    81,    -1,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    -1,
      -1,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     113,   114,    -1,    43,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     3,    -1,    56,    -1,    58,    59,
      -1,    -1,    -1,    -1,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    -1,    79,
      80,    81,    -1,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    -1,    -1,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,     3,    -1,    56,    -1,    58,
      59,    -1,    -1,   113,   114,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    -1,    -1,    78,
      79,    80,    81,    -1,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    -1,    -1,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,     3,    -1,    56,    -1,
      58,    59,    -1,    -1,   113,   114,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    -1,    -1,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,     3,    -1,    56,
      -1,    58,    59,    -1,    -1,   113,   114,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    -1,
      -1,    -1,    79,    80,    81,    -1,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    -1,    -1,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,     3,    -1,
      56,    -1,    58,    59,    -1,    -1,   113,   114,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    -1,    -1,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,     3,
      -1,    56,    -1,    58,    59,    -1,    -1,   113,   114,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    -1,    -1,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
       3,    -1,    56,    -1,    58,    59,    -1,    -1,   113,   114,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    -1,    -1,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,     3,    -1,    56,    -1,    58,    59,    -1,    -1,   113,
     114,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    -1,
      -1,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,     3,    -1,    56,    -1,    58,    59,    -1,    -1,
     113,   114,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      -1,    -1,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,    -1,    -1,    56,    -1,    58,    59,    -1,
      -1,   113,   114,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    -1,    -1,    -1,    79,    80,
      81,    -1,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    -1,    -1,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,     4,     5,     6,     7,     8,    -1,
      -1,    -1,   113,   114,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    43,    44,    -1,    46,    -1,    48,    -1,
      -1,    51,    52,    -1,    -1,    -1,    56,    57,    58,    -1,
      60,    -1,    62,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,     5,     6,
       7,     8,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    26,
      27,    -1,    -1,    -1,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,    43,    44,    -1,    46,
      -1,    48,    -1,    -1,    51,    52,    -1,    -1,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,
      97,    26,    27,    -1,   101,    -1,    -1,    -1,   105,   106,
     107,   108,   109,   110,    -1,   112,   113,   114,   115,    44,
      -1,    46,    -1,    48,    -1,    -1,    51,    52,    -1,    -1,
      -1,    56,    57,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    26,    27,    -1,    -1,    -1,    -1,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,    44,    -1,    46,    -1,    48,    -1,    -1,    51,    52,
      -1,    -1,    -1,    56,    57,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    26,    27,    -1,    -1,    -1,
      -1,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,    44,    -1,    46,    -1,    48,    -1,    -1,
      51,    52,    -1,    -1,    55,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    92,    93,    -1,    -1,    -1,    97,    26,    27,    -1,
      -1,    -1,    -1,    -1,   105,   106,   107,   108,   109,   110,
      -1,   112,   113,   114,   115,    44,    -1,    46,    -1,    48,
      -1,    -1,    51,    52,    -1,    -1,    -1,    56,    57,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,    26,
      27,    -1,    -1,    -1,    -1,    -1,   105,   106,   107,   108,
     109,   110,    -1,   112,   113,   114,   115,    44,    -1,    46,
      -1,    48,    -1,    -1,    51,    52,    -1,    -1,    55,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,
      97,    26,    27,    -1,    -1,    -1,    -1,    -1,   105,   106,
     107,   108,   109,   110,    -1,   112,   113,   114,   115,    44,
      -1,    46,    -1,    48,    -1,    -1,    51,    52,    -1,    -1,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,    43,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,    56,    -1,    58,    59,    -1,    -1,    62,    -1,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    -1,    -1,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
      43,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   113,   114,
      -1,    -1,    -1,    56,    57,    58,    59,    -1,    -1,    -1,
      -1,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    -1,
      -1,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,    43,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     113,   114,    -1,    -1,    -1,    56,    57,    58,    59,    -1,
      -1,    -1,    -1,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    -1,    -1,    -1,    79,    80,
      81,    -1,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    -1,    -1,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,    -1,    55,    56,    -1,    58,    59,
      -1,    -1,   113,   114,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    -1,    79,
      80,    81,    -1,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    -1,    -1,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,    -1,    -1,    56,    57,    58,
      59,    -1,    -1,   113,   114,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    -1,    -1,    -1,
      79,    80,    81,    -1,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    -1,    -1,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,    -1,    -1,    56,    -1,
      58,    59,    -1,    -1,   113,   114,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    77,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    -1,    -1,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,    -1,    -1,    -1,
      -1,    56,    -1,    58,    59,   113,   114,    62,    -1,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    -1,    -1,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
      -1,    -1,    56,    57,    58,    59,    -1,    -1,   113,   114,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    -1,    -1,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,    -1,    -1,    56,    57,    58,    59,    -1,    -1,   113,
     114,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    -1,
      -1,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,    -1,    -1,    56,    -1,    58,    59,    -1,    -1,
     113,   114,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      -1,    -1,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,    56,    -1,    58,    59,    -1,    -1,    -1,
      -1,   113,   114,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    75,    -1,    -1,    78,    79,    80,    81,    -1,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    -1,
      -1,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,    56,    -1,    58,    59,    -1,    -1,    -1,    -1,
     113,   114,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    -1,    -1,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,    56,    -1,    58,    59,    -1,    -1,    -1,    -1,   113,
     114,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    80,    81,    -1,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    -1,    -1,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
      56,    -1,    58,    59,    -1,    -1,    -1,    -1,   113,   114,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    -1,    -1,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,    56,
      -1,    58,    59,    -1,    -1,    -1,    -1,   113,   114,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    -1,    -1,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   113,   114
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   117,   118,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    30,    31,    33,    34,    35,    37,
      38,    39,    40,    41,    42,    44,    46,    47,    48,    49,
      50,    51,    52,    54,    56,    58,    60,    82,    86,    92,
      93,    97,   103,   105,   106,   107,   108,   109,   110,   112,
     113,   114,   115,   119,   120,   122,   123,   124,   126,   127,
     129,   130,   131,   134,   136,   137,   145,   146,   147,   155,
     156,   157,   164,   166,   179,   181,   190,   192,   199,   200,
     201,   203,   205,   213,   214,   215,   217,   218,   220,   223,
     241,   246,   251,   255,   256,   258,   259,   261,   262,   263,
     265,   267,   271,   273,   274,   275,   276,   277,     3,    43,
      63,     3,     1,     6,   125,   258,     1,     6,    44,   261,
       1,     3,     1,     3,    14,     1,   261,     1,   258,   280,
       1,   261,     1,     1,   261,     1,     3,    43,     1,   261,
       1,     6,     1,     6,     1,     3,   261,   252,     1,     6,
       7,     1,   261,   263,     1,     6,     1,     3,     6,   216,
       1,     6,    32,   219,     1,     6,   221,   222,     1,     6,
     266,   272,     1,   261,    57,   261,   278,     1,     3,    43,
       6,   261,    43,    57,    62,   261,   277,   281,   268,   261,
       1,     3,   261,   277,   261,   261,   261,     1,     3,   277,
     261,   261,   261,   261,   261,   261,     4,     6,    26,    59,
     261,   261,     4,   258,   128,    31,    33,    44,   123,   135,
     123,   165,   180,   191,    45,   208,   211,   212,   123,     1,
      56,     3,    56,    58,    59,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    79,    80,    81,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     113,   114,   262,    75,    78,     1,     4,     5,     7,     8,
      51,    52,    97,   121,   257,   261,     3,     3,    78,    75,
       3,    43,     3,    43,     3,     3,     3,     3,    43,     3,
      43,     3,    75,    78,    91,     3,     3,     3,     3,     3,
       3,   123,     3,     3,     3,   224,     3,   247,     3,     3,
       1,     6,   253,   254,     3,     3,     3,     3,     3,     3,
      75,     3,     3,    78,     3,     1,     6,     7,     1,     3,
      32,    78,     3,    75,     3,    78,     3,     1,    56,   269,
     269,     3,     3,     1,    57,    78,   279,     3,   132,   123,
     242,    55,    57,   261,    57,    43,    62,     1,    57,     1,
      57,    78,     1,     6,   206,   207,   270,     3,     3,     3,
       3,     4,     6,    26,   144,   144,   167,   182,   144,     1,
       3,    43,   144,   209,   210,     3,     1,   206,    55,   277,
     101,   261,     6,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,     1,   261,   261,   261,   261,
     261,   261,   261,   261,   261,     6,   261,   261,     3,   260,
     260,   260,   260,   260,   260,   260,   260,   260,   260,   260,
     261,   261,     3,     4,     3,   258,   261,   151,   261,   258,
       1,   261,     1,    56,   225,   226,     1,    32,   228,   248,
       3,    78,     1,     1,   257,     6,     3,     3,    76,     3,
      76,     3,     6,     7,     6,     6,     7,   121,   222,     3,
     206,   208,   208,   261,   144,     3,    57,    57,   261,   261,
      57,   261,    62,     1,    62,    78,   208,     9,   123,    16,
      17,   138,   140,   141,   143,     1,     3,    23,    24,   168,
     173,   175,     1,     3,    23,   173,   183,    29,   193,   194,
     195,   196,     3,    43,     9,   144,   123,   204,     1,    55,
       1,    55,   261,    57,    78,     1,    43,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,     3,    78,
      75,   149,    77,     3,   148,     3,   206,   232,   228,     3,
       6,   229,   230,     3,   249,     1,   254,     3,     3,     6,
       6,     3,    76,    91,     3,    76,    91,     1,    55,   144,
     144,     9,   243,    43,    57,    62,   207,   144,     3,     1,
       3,     1,   261,     9,   139,   142,     3,     1,     6,     7,
       8,   121,   177,   178,     1,     9,   174,     3,     1,     4,
       6,   188,   189,     9,     1,     3,     4,     6,    91,   197,
     198,     9,   195,   144,     3,     9,    55,   202,     3,    43,
     264,    57,   277,     1,   261,   277,   261,     3,    43,   150,
       1,   261,   150,     1,    55,     3,     6,     9,    36,    44,
      45,    53,   200,   218,   233,   234,   236,   237,   238,     3,
      56,   231,    78,     3,     9,   200,   218,   234,   236,   250,
       3,     3,     6,     6,     6,     6,     3,     9,     9,   133,
     261,     3,     6,     9,   218,   244,   261,   261,    61,     3,
       3,     3,     3,   144,   144,     3,    43,    77,     3,    43,
      78,     3,     3,    43,   176,     3,    43,     3,    43,    78,
       3,     3,   258,     3,    78,    91,     3,     3,    43,    55,
      55,     3,    19,    20,    21,   123,   153,   154,   158,   160,
     162,   123,     1,    78,   152,   227,    75,     3,     3,     6,
       6,    36,   239,    55,   277,   230,     3,     3,     3,     3,
       3,     3,     3,    75,    78,   245,     3,    57,   138,   171,
     172,   121,   169,   170,   178,   144,   123,   186,   187,   184,
     185,   189,     3,   198,   258,     3,     1,     3,    43,     1,
       3,    43,     1,     3,    43,     9,   153,     1,   261,    55,
     261,   235,    75,     3,     3,     3,     9,   200,   240,    55,
     257,   144,   123,   144,   123,   144,   123,   144,   123,     3,
       3,   159,   123,     3,   161,   123,     3,   163,   123,     3,
       3,   208,   257,     3,   245,   144,   144,   144,   144,     3,
       9,     9,     9,     9,     3,     3,     3,     3
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (&yylval, YYLEX_PARAM)
#else
# define YYLEX yylex (&yylval)
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void
yy_stack_print (yybottom, yytop)
    yytype_int16 *yybottom;
    yytype_int16 *yytop;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}

/* Prevent warnings from -Wmissing-prototypes.  */
#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */





/*-------------------------.
| yyparse or yypush_parse.  |
`-------------------------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

    /* Number of syntax errors so far.  */
    int yynerrs;

    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       `yyss': related to states.
       `yyvs': related to semantic values.

       Refer to the stacks thru separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yytoken = 0;
  yyss = yyssa;
  yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */
  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;

	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),
		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss_alloc, yyss);
	YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 6:

/* Line 1455 of yacc.c  */
#line 207 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); }
    break;

  case 7:

/* Line 1455 of yacc.c  */
#line 208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); }
    break;

  case 8:

/* Line 1455 of yacc.c  */
#line 212 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat)=0; }
    break;

  case 10:

/* Line 1455 of yacc.c  */
#line 215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 11:

/* Line 1455 of yacc.c  */
#line 220 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 12:

/* Line 1455 of yacc.c  */
#line 225 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 13:

/* Line 1455 of yacc.c  */
#line 230 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 14:

/* Line 1455 of yacc.c  */
#line 235 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 19:

/* Line 1455 of yacc.c  */
#line 246 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.integer) = - (yyvsp[(2) - (2)].integer); }
    break;

  case 20:

/* Line 1455 of yacc.c  */
#line 251 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *(yyvsp[(2) - (3)].stringp), false );
      }
    break;

  case 21:

/* Line 1455 of yacc.c  */
#line 257 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *(yyvsp[(2) - (3)].stringp), true );
      }
    break;

  case 22:

/* Line 1455 of yacc.c  */
#line 263 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
      }
    break;

  case 23:

/* Line 1455 of yacc.c  */
#line 269 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); }
    break;

  case 24:

/* Line 1455 of yacc.c  */
#line 270 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_stat)=0;}
    break;

  case 25:

/* Line 1455 of yacc.c  */
#line 271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; }
    break;

  case 26:

/* Line 1455 of yacc.c  */
#line 272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; }
    break;

  case 27:

/* Line 1455 of yacc.c  */
#line 273 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; }
    break;

  case 28:

/* Line 1455 of yacc.c  */
#line 274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; }
    break;

  case 29:

/* Line 1455 of yacc.c  */
#line 275 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;}
    break;

  case 30:

/* Line 1455 of yacc.c  */
#line 280 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (! COMPILER->isInteractive()) && ((!(yyvsp[(1) - (2)].fal_val)->isExpr()) || (!(yyvsp[(1) - (2)].fal_val)->asExpr()->isStandAlone()) ) )
         {
            Falcon::StmtFunction *func = COMPILER->getFunctionContext();
            if ( func == 0 || ! func->isLambda() )
               COMPILER->raiseError(Falcon::e_noeffect );
         }

         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) );
      }
    break;

  case 31:

/* Line 1455 of yacc.c  */
#line 291 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (4)].fal_adecl) );
         COMPILER->defineVal( first );
         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, (yyvsp[(3) - (4)].fal_val) ) ) );
      }
    break;

  case 32:

/* Line 1455 of yacc.c  */
#line 297 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (6)].fal_adecl)->size() != (yyvsp[(5) - (6)].fal_adecl)->size() + 1 )
         {
            COMPILER->raiseError(Falcon::e_unpack_size );
         }
         Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (6)].fal_adecl) );

         COMPILER->defineVal( first );
         (yyvsp[(5) - (6)].fal_adecl)->pushFront( (yyvsp[(3) - (6)].fal_val) );
         Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (6)].fal_adecl) );
         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, second ) ) );
      }
    break;

  case 50:

/* Line 1455 of yacc.c  */
#line 331 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_def );
   }
    break;

  case 51:

/* Line 1455 of yacc.c  */
#line 334 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true );
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr( CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ) ) );
   }
    break;

  case 52:

/* Line 1455 of yacc.c  */
#line 340 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr(CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) ) ) );
   }
    break;

  case 53:

/* Line 1455 of yacc.c  */
#line 349 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; }
    break;

  case 54:

/* Line 1455 of yacc.c  */
#line 351 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); }
    break;

  case 55:

/* Line 1455 of yacc.c  */
#line 355 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
    break;

  case 56:

/* Line 1455 of yacc.c  */
#line 362 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      }
    break;

  case 57:

/* Line 1455 of yacc.c  */
#line 369 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      }
    break;

  case 58:

/* Line 1455 of yacc.c  */
#line 377 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 59:

/* Line 1455 of yacc.c  */
#line 378 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; }
    break;

  case 60:

/* Line 1455 of yacc.c  */
#line 382 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 61:

/* Line 1455 of yacc.c  */
#line 383 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 62:

/* Line 1455 of yacc.c  */
#line 387 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = new Falcon::StmtLoop( LINE );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
    break;

  case 63:

/* Line 1455 of yacc.c  */
#line 394 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = static_cast<Falcon::StmtLoop* >(COMPILER->getContext());
         w->setCondition((yyvsp[(6) - (7)].fal_val));
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      }
    break;

  case 64:

/* Line 1455 of yacc.c  */
#line 402 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = new Falcon::StmtLoop( LINE );
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
         (yyval.fal_stat) = w;
      }
    break;

  case 65:

/* Line 1455 of yacc.c  */
#line 408 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_loop );
      (yyval.fal_stat) = 0;
   }
    break;

  case 66:

/* Line 1455 of yacc.c  */
#line 415 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 67:

/* Line 1455 of yacc.c  */
#line 416 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(1) - (1)].fal_val); }
    break;

  case 68:

/* Line 1455 of yacc.c  */
#line 420 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      }
    break;

  case 69:

/* Line 1455 of yacc.c  */
#line 428 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 70:

/* Line 1455 of yacc.c  */
#line 435 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      }
    break;

  case 71:

/* Line 1455 of yacc.c  */
#line 445 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 72:

/* Line 1455 of yacc.c  */
#line 446 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; }
    break;

  case 73:

/* Line 1455 of yacc.c  */
#line 450 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 74:

/* Line 1455 of yacc.c  */
#line 451 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 77:

/* Line 1455 of yacc.c  */
#line 458 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      }
    break;

  case 80:

/* Line 1455 of yacc.c  */
#line 468 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); }
    break;

  case 81:

/* Line 1455 of yacc.c  */
#line 473 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      }
    break;

  case 83:

/* Line 1455 of yacc.c  */
#line 485 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 84:

/* Line 1455 of yacc.c  */
#line 486 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; }
    break;

  case 86:

/* Line 1455 of yacc.c  */
#line 491 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   }
    break;

  case 87:

/* Line 1455 of yacc.c  */
#line 498 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      }
    break;

  case 88:

/* Line 1455 of yacc.c  */
#line 507 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      }
    break;

  case 89:

/* Line 1455 of yacc.c  */
#line 515 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      }
    break;

  case 90:

/* Line 1455 of yacc.c  */
#line 525 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      }
    break;

  case 91:

/* Line 1455 of yacc.c  */
#line 534 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      }
    break;

  case 92:

/* Line 1455 of yacc.c  */
#line 543 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = (yyvsp[(2) - (4)].fal_adecl);
         f = new Falcon::StmtForin( LINE, decl, (yyvsp[(4) - (4)].fal_val) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      }
    break;

  case 93:

/* Line 1455 of yacc.c  */
#line 554 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      }
    break;

  case 94:

/* Line 1455 of yacc.c  */
#line 563 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f;
         COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) );
         Falcon::ArrayDecl *decl = new Falcon::ArrayDecl();
         decl->pushBack( (yyvsp[(2) - (4)].fal_val) );
         f = new Falcon::StmtForin( LINE, decl, (yyvsp[(4) - (4)].fal_val) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      }
    break;

  case 95:

/* Line 1455 of yacc.c  */
#line 576 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      }
    break;

  case 96:

/* Line 1455 of yacc.c  */
#line 586 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { delete (yyvsp[(2) - (5)].fal_adecl);
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 97:

/* Line 1455 of yacc.c  */
#line 591 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 98:

/* Line 1455 of yacc.c  */
#line 599 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      }
    break;

  case 100:

/* Line 1455 of yacc.c  */
#line 612 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::RangeDecl* rd = new Falcon::RangeDecl( (yyvsp[(1) - (4)].fal_val),
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_oob, (yyvsp[(3) - (4)].fal_val))), (yyvsp[(4) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( rd );
      }
    break;

  case 101:

/* Line 1455 of yacc.c  */
#line 618 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val), 0 ) );
      }
    break;

  case 102:

/* Line 1455 of yacc.c  */
#line 622 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(1) - (3)].fal_val), 0, 0 ) );
      }
    break;

  case 103:

/* Line 1455 of yacc.c  */
#line 628 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 104:

/* Line 1455 of yacc.c  */
#line 629 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 105:

/* Line 1455 of yacc.c  */
#line 630 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 108:

/* Line 1455 of yacc.c  */
#line 639 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      }
    break;

  case 112:

/* Line 1455 of yacc.c  */
#line 653 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getLoop());
         if ( f == 0 || f->type() != Falcon::Statement::t_forin )
         {
            COMPILER->raiseError( Falcon::e_syn_fordot );
            delete (yyvsp[(2) - (3)].fal_val);
            (yyval.fal_stat) = 0;
         }
         else {
            (yyval.fal_stat) = new Falcon::StmtFordot( LINE, (yyvsp[(2) - (3)].fal_val) );
         }
      }
    break;

  case 113:

/* Line 1455 of yacc.c  */
#line 666 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      }
    break;

  case 114:

/* Line 1455 of yacc.c  */
#line 674 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 115:

/* Line 1455 of yacc.c  */
#line 678 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 116:

/* Line 1455 of yacc.c  */
#line 684 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 117:

/* Line 1455 of yacc.c  */
#line 690 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      }
    break;

  case 118:

/* Line 1455 of yacc.c  */
#line 697 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 119:

/* Line 1455 of yacc.c  */
#line 702 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 120:

/* Line 1455 of yacc.c  */
#line 711 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   }
    break;

  case 121:

/* Line 1455 of yacc.c  */
#line 720 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         COMPILER->pushContextSet( &f->firstBlock() );
		 // Push anyhow an empty item, that is needed for to check again for thio blosk
		 f->firstBlock().push_back( new Falcon::StmtNone( LINE ) );
      }
    break;

  case 122:

/* Line 1455 of yacc.c  */
#line 732 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 123:

/* Line 1455 of yacc.c  */
#line 734 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->firstBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 124:

/* Line 1455 of yacc.c  */
#line 743 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); }
    break;

  case 125:

/* Line 1455 of yacc.c  */
#line 747 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 f->lastBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->lastBlock() );
      }
    break;

  case 126:

/* Line 1455 of yacc.c  */
#line 759 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 127:

/* Line 1455 of yacc.c  */
#line 760 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->lastBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 128:

/* Line 1455 of yacc.c  */
#line 769 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); }
    break;

  case 129:

/* Line 1455 of yacc.c  */
#line 773 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 // Apparently you get a segfault without it.
		 // (Note that the formiddle: version below does *not* need it
		 f->middleBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->middleBlock() );
      }
    break;

  case 130:

/* Line 1455 of yacc.c  */
#line 787 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 131:

/* Line 1455 of yacc.c  */
#line 789 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->middleBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 132:

/* Line 1455 of yacc.c  */
#line 798 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_formiddle ); }
    break;

  case 133:

/* Line 1455 of yacc.c  */
#line 802 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 134:

/* Line 1455 of yacc.c  */
#line 810 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 135:

/* Line 1455 of yacc.c  */
#line 819 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 136:

/* Line 1455 of yacc.c  */
#line 821 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 139:

/* Line 1455 of yacc.c  */
#line 830 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); }
    break;

  case 141:

/* Line 1455 of yacc.c  */
#line 836 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 143:

/* Line 1455 of yacc.c  */
#line 846 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 144:

/* Line 1455 of yacc.c  */
#line 854 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 145:

/* Line 1455 of yacc.c  */
#line 858 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 147:

/* Line 1455 of yacc.c  */
#line 870 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 148:

/* Line 1455 of yacc.c  */
#line 880 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 150:

/* Line 1455 of yacc.c  */
#line 889 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();
         if ( ! stmt->defaultBlock().empty() )
         {
            COMPILER->raiseError(Falcon::e_switch_default, "", CURRENT_LINE );
         }
         COMPILER->pushContextSet( &stmt->defaultBlock() );
      }
    break;

  case 154:

/* Line 1455 of yacc.c  */
#line 903 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); }
    break;

  case 156:

/* Line 1455 of yacc.c  */
#line 907 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      }
    break;

  case 159:

/* Line 1455 of yacc.c  */
#line 919 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      }
    break;

  case 160:

/* Line 1455 of yacc.c  */
#line 928 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 161:

/* Line 1455 of yacc.c  */
#line 940 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].stringp) );
         if ( ! stmt->addStringCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 162:

/* Line 1455 of yacc.c  */
#line 951 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (yyvsp[(1) - (3)].integer) ), new Falcon::Value( (yyvsp[(3) - (3)].integer) ) ) );
         if ( ! stmt->addRangeCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 163:

/* Line 1455 of yacc.c  */
#line 962 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 164:

/* Line 1455 of yacc.c  */
#line 982 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 165:

/* Line 1455 of yacc.c  */
#line 990 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 166:

/* Line 1455 of yacc.c  */
#line 999 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 167:

/* Line 1455 of yacc.c  */
#line 1001 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 170:

/* Line 1455 of yacc.c  */
#line 1010 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); }
    break;

  case 172:

/* Line 1455 of yacc.c  */
#line 1016 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 174:

/* Line 1455 of yacc.c  */
#line 1026 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 175:

/* Line 1455 of yacc.c  */
#line 1035 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 176:

/* Line 1455 of yacc.c  */
#line 1039 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 178:

/* Line 1455 of yacc.c  */
#line 1051 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 179:

/* Line 1455 of yacc.c  */
#line 1061 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 183:

/* Line 1455 of yacc.c  */
#line 1075 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 184:

/* Line 1455 of yacc.c  */
#line 1087 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 185:

/* Line 1455 of yacc.c  */
#line 1107 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   }
    break;

  case 186:

/* Line 1455 of yacc.c  */
#line 1114 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      }
    break;

  case 187:

/* Line 1455 of yacc.c  */
#line 1124 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      }
    break;

  case 189:

/* Line 1455 of yacc.c  */
#line 1133 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); }
    break;

  case 195:

/* Line 1455 of yacc.c  */
#line 1153 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }
         // but continue by pushing this new context
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      }
    break;

  case 196:

/* Line 1455 of yacc.c  */
#line 1171 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }

         // but continue by pushing this new context
         COMPILER->defineVal( (yyvsp[(3) - (4)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(3) - (4)].fal_val) );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      }
    break;

  case 197:

/* Line 1455 of yacc.c  */
#line 1191 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 198:

/* Line 1455 of yacc.c  */
#line 1201 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 199:

/* Line 1455 of yacc.c  */
#line 1212 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   }
    break;

  case 202:

/* Line 1455 of yacc.c  */
#line 1225 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );

         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 203:

/* Line 1455 of yacc.c  */
#line 1237 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 204:

/* Line 1455 of yacc.c  */
#line 1259 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 205:

/* Line 1455 of yacc.c  */
#line 1260 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; }
    break;

  case 206:

/* Line 1455 of yacc.c  */
#line 1272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 207:

/* Line 1455 of yacc.c  */
#line 1278 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 209:

/* Line 1455 of yacc.c  */
#line 1287 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 210:

/* Line 1455 of yacc.c  */
#line 1288 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 211:

/* Line 1455 of yacc.c  */
#line 1291 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); }
    break;

  case 213:

/* Line 1455 of yacc.c  */
#line 1296 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 214:

/* Line 1455 of yacc.c  */
#line 1297 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 215:

/* Line 1455 of yacc.c  */
#line 1304 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // if we are in a class, I have to create the symbol classname.functionname
         Falcon::Statement *parent = COMPILER->getContext();
         Falcon::String *func_name;
         if ( parent != 0 && parent->type() == Falcon::Statement::t_class ) {
            Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
            Falcon::String complete_name = stmt_cls->symbol()->name() + "." + *(yyvsp[(2) - (2)].stringp);
            func_name = COMPILER->addString( complete_name );
         }
         else if ( parent != 0 && parent->type() == Falcon::Statement::t_state ) 
         {
            Falcon::StmtState *stmt_state = static_cast< Falcon::StmtState *>( parent );
            Falcon::String complete_name =  
                  stmt_state->owner()->symbol()->name() + "." + 
                  * stmt_state->name() + "$" + *(yyvsp[(2) - (2)].stringp);
                  
            func_name = COMPILER->addString( complete_name );
         }
         else
            func_name = (yyvsp[(2) - (2)].stringp);

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( func_name );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( func_name );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         // and eventually add it as a class property
         if ( parent != 0 )
         {
            if( parent->type() == Falcon::Statement::t_class ) 
            {
               Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
               Falcon::ClassDef *cd = stmt_cls->symbol()->getClassDef();
               if ( cd->hasProperty( *(yyvsp[(2) - (2)].stringp) ) ) {
                  COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (2)].stringp) );
               }
               else 
               {
                   cd->addProperty( (yyvsp[(2) - (2)].stringp), new Falcon::VarDef( sym ) );
                   // is this a setter/getter?
                   if( ( (yyvsp[(2) - (2)].stringp)->find( "__set_" ) == 0 || (yyvsp[(2) - (2)].stringp)->find( "__get_" ) == 0 ) && (yyvsp[(2) - (2)].stringp)->length() > 6 )
                   {
                      Falcon::String *pname = COMPILER->addString( (yyvsp[(2) - (2)].stringp)->subString( 6 ));
                      Falcon::VarDef *pd = cd->getProperty( *pname );
                      if( pd == 0 )
                      {
                        pd = new Falcon::VarDef;
                        cd->addProperty( pname, pd );
                        pd->setReflective( Falcon::e_reflectSetGet, 0xFFFFFFFF );
                      }
                      else if( ! pd->isReflective() )
                      {
                        COMPILER->raiseError(Falcon::e_prop_adef, *pname );
                      }
   
                   }
               }
            }
            else if ( parent->type() == Falcon::Statement::t_state ) 
            {
   
               Falcon::StmtState *stmt_state = static_cast< Falcon::StmtState *>( parent );            
               if( ! stmt_state->addFunction( (yyvsp[(2) - (2)].stringp), sym ) )
               {
                  COMPILER->raiseError(Falcon::e_sm_adef, *(yyvsp[(2) - (2)].stringp) );
               }
               
               // eventually add a property where to store this thing
               Falcon::ClassDef *cd = stmt_state->owner()->symbol()->getClassDef();
               if ( ! cd->hasProperty( *(yyvsp[(2) - (2)].stringp) ) )
                  cd->addProperty( (yyvsp[(2) - (2)].stringp), new Falcon::VarDef );
            }
         }
         
         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      }
    break;

  case 219:

/* Line 1455 of yacc.c  */
#line 1410 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym != 0 ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }
         else {
            Falcon::FuncDef *func = COMPILER->getFunction();
            Falcon::Symbol *sym = new Falcon::Symbol( COMPILER->module(), (yyvsp[(1) - (1)].stringp) );
            COMPILER->module()->addSymbol( sym );
            func->addParameter( sym );
         }
      }
    break;

  case 221:

/* Line 1455 of yacc.c  */
#line 1427 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 222:

/* Line 1455 of yacc.c  */
#line 1433 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 223:

/* Line 1455 of yacc.c  */
#line 1438 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 224:

/* Line 1455 of yacc.c  */
#line 1444 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 226:

/* Line 1455 of yacc.c  */
#line 1453 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); }
    break;

  case 228:

/* Line 1455 of yacc.c  */
#line 1458 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); }
    break;

  case 229:

/* Line 1455 of yacc.c  */
#line 1468 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      }
    break;

  case 230:

/* Line 1455 of yacc.c  */
#line 1471 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; }
    break;

  case 231:

/* Line 1455 of yacc.c  */
#line 1480 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // TODO: evalute const expressions on the fly.
         Falcon::Value *val = (yyvsp[(4) - (5)].fal_val); //COMPILER->exprSimplify( $4 );
         // will raise an error in case the expression is not atomic.
         COMPILER->addConstant( *(yyvsp[(2) - (5)].stringp), val, LINE );
         // we don't need the expression anymore
         // no other action:
         (yyval.fal_stat) = 0;
      }
    break;

  case 232:

/* Line 1455 of yacc.c  */
#line 1490 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      }
    break;

  case 233:

/* Line 1455 of yacc.c  */
#line 1495 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      }
    break;

  case 234:

/* Line 1455 of yacc.c  */
#line 1507 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 235:

/* Line 1455 of yacc.c  */
#line 1516 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      }
    break;

  case 236:

/* Line 1455 of yacc.c  */
#line 1523 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      }
    break;

  case 237:

/* Line 1455 of yacc.c  */
#line 1531 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      }
    break;

  case 238:

/* Line 1455 of yacc.c  */
#line 1536 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      }
    break;

  case 239:

/* Line 1455 of yacc.c  */
#line 1544 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (3)].fal_genericList) );
         (yyval.fal_stat) = 0;
      }
    break;

  case 240:

/* Line 1455 of yacc.c  */
#line 1549 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (5)].fal_genericList), (yyvsp[(4) - (5)].stringp), 0, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 241:

/* Line 1455 of yacc.c  */
#line 1554 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (5)].fal_genericList), (yyvsp[(4) - (5)].stringp), 0, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 242:

/* Line 1455 of yacc.c  */
#line 1559 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (7)].fal_genericList)->begin();
         int counter = 0;
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            if ( counter == 0 )
               COMPILER->importAlias( symName, (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), false );
            delete symName;
            li = li->next();
            counter++;
         }
         delete (yyvsp[(2) - (7)].fal_genericList);

         if ( counter != 1 )
            COMPILER->raiseError(Falcon::e_syn_import );

         (yyval.fal_stat) = 0;
      }
    break;

  case 243:

/* Line 1455 of yacc.c  */
#line 1579 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (7)].fal_genericList)->begin();
         int counter = 0;
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            if ( counter == 0 )
               COMPILER->importAlias( symName, (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), true );
            delete symName;
            li = li->next();
            counter++;
         }
         delete (yyvsp[(2) - (7)].fal_genericList);

         if ( counter != 1 )
            COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 244:

/* Line 1455 of yacc.c  */
#line 1598 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 245:

/* Line 1455 of yacc.c  */
#line 1603 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 246:

/* Line 1455 of yacc.c  */
#line 1608 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 247:

/* Line 1455 of yacc.c  */
#line 1613 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (4)].fal_genericList)->begin();
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            delete symName;
            li = li->next();
         }
         delete (yyvsp[(2) - (4)].fal_genericList);

         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 248:

/* Line 1455 of yacc.c  */
#line 1627 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 249:

/* Line 1455 of yacc.c  */
#line 1632 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 250:

/* Line 1455 of yacc.c  */
#line 1637 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 251:

/* Line 1455 of yacc.c  */
#line 1642 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 252:

/* Line 1455 of yacc.c  */
#line 1647 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 253:

/* Line 1455 of yacc.c  */
#line 1656 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addAttribute( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val), LINE );
     }
    break;

  case 254:

/* Line 1455 of yacc.c  */
#line 1661 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_attrdecl );
     }
    break;

  case 255:

/* Line 1455 of yacc.c  */
#line 1668 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::List *lst = new Falcon::List;
         lst->pushBack( new Falcon::String( *(yyvsp[(1) - (1)].stringp) ) );
         (yyval.fal_genericList) = lst;
      }
    break;

  case 256:

/* Line 1455 of yacc.c  */
#line 1674 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(1) - (3)].fal_genericList)->pushBack( new Falcon::String( *(yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_genericList) = (yyvsp[(1) - (3)].fal_genericList);
      }
    break;

  case 257:

/* Line 1455 of yacc.c  */
#line 1686 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 258:

/* Line 1455 of yacc.c  */
#line 1691 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_directive );
         (yyval.fal_stat)=0;
     }
    break;

  case 261:

/* Line 1455 of yacc.c  */
#line 1704 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 262:

/* Line 1455 of yacc.c  */
#line 1708 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 263:

/* Line 1455 of yacc.c  */
#line 1712 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), (yyvsp[(3) - (3)].integer) );
      }
    break;

  case 264:

/* Line 1455 of yacc.c  */
#line 1725 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setClass( def );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         // We don't have a context set here
         COMPILER->pushFunction( def );
      }
    break;

  case 265:

/* Line 1455 of yacc.c  */
#line 1757 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         // check for expressions in from clauses
         COMPILER->checkLocalUndefined();

         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls != 0 )
         {
            if ( cls->ctorFunction() == 0  )
            {
               Falcon::ClassDef *cd = cls->symbol()->getClassDef();
               if ( cd->inheritance().size() != 0 )
               {
                  COMPILER->buildCtorFor( cls );
                  // COMPILER->addStatement( func ); should be done in buildCtorFor
                  // cls->ctorFunction( func ); idem
               }
            }
            COMPILER->popContext();
            //We didn't pushed a context set
            COMPILER->popFunction();
         }
      }
    break;

  case 267:

/* Line 1455 of yacc.c  */
#line 1791 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      }
    break;

  case 270:

/* Line 1455 of yacc.c  */
#line 1799 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 271:

/* Line 1455 of yacc.c  */
#line 1800 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 276:

/* Line 1455 of yacc.c  */
#line 1817 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         // creates or find the symbol.
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol((yyvsp[(1) - (2)].stringp));
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         Falcon::InheritDef *idef = new Falcon::InheritDef(sym);

         if ( clsdef->addInheritance( idef ) )
         {
            cls->addInitExpression(
               new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_inherit,
                     new Falcon::Value( sym ), (yyvsp[(2) - (2)].fal_val) ) ) );
         }
         else {
            COMPILER->raiseError(Falcon::e_prop_adef );
            delete idef;
            delete (yyvsp[(2) - (2)].fal_val);
         }
      }
    break;

  case 277:

/* Line 1455 of yacc.c  */
#line 1840 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; }
    break;

  case 278:

/* Line 1455 of yacc.c  */
#line 1841 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 279:

/* Line 1455 of yacc.c  */
#line 1843 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = (yyvsp[(2) - (3)].fal_adecl) == 0 ? 0 : new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
   }
    break;

  case 283:

/* Line 1455 of yacc.c  */
#line 1856 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 284:

/* Line 1455 of yacc.c  */
#line 1859 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   }
    break;

  case 285:

/* Line 1455 of yacc.c  */
#line 1879 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );         
         if( ! cls->addState( static_cast<Falcon::StmtState *>((yyvsp[(1) - (1)].fal_stat)) ) )
         {
            const Falcon::String* name = static_cast<Falcon::StmtState *>((yyvsp[(1) - (1)].fal_stat))->name();
            delete (yyvsp[(1) - (1)].fal_stat);
            COMPILER->raiseError( Falcon::e_state_adef, *name );
            
         }
      }
    break;

  case 288:

/* Line 1455 of yacc.c  */
#line 1895 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         if( cls->initGiven() ) {
            COMPILER->raiseError(Falcon::e_init_given );
         }
         else
         {
            cls->initGiven( true );
            Falcon::StmtFunction *func = cls->ctorFunction();
            if ( func == 0 ) {
               func = COMPILER->buildCtorFor( cls );
            }

            // prepare the statement allocation context
            COMPILER->pushContext( func );
            COMPILER->pushFunctionContext( func );
            COMPILER->pushContextSet( &func->statements() );
            COMPILER->pushFunction( func->symbol()->getFuncDef() );
         }
      }
    break;

  case 289:

/* Line 1455 of yacc.c  */
#line 1920 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
         COMPILER->popFunctionContext();
      }
    break;

  case 290:

/* Line 1455 of yacc.c  */
#line 1930 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(4) - (5)].fal_val)->genVarDef();

      if ( def != 0 ) {
         Falcon::String prop_name = cls->symbol()->name() + "." + *(yyvsp[(2) - (5)].stringp);
         Falcon::Symbol *sym = COMPILER->addGlobalVar( COMPILER->addString(prop_name), def );
         if( clsdef->hasProperty( *(yyvsp[(2) - (5)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (5)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(2) - (5)].stringp), new Falcon::VarDef( Falcon::VarDef::t_reference, sym) );
      }
      else {
         COMPILER->raiseError(Falcon::e_static_const );
      }
      delete (yyvsp[(4) - (5)].fal_val); // the expression is not needed anymore
      (yyval.fal_stat) = 0; // we don't add any statement
   }
    break;

  case 291:

/* Line 1455 of yacc.c  */
#line 1952 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(3) - (4)].fal_val)->genVarDef();

      if ( def != 0 ) {
         if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), def );
         delete (yyvsp[(3) - (4)].fal_val); // the expression is not needed anymore
         (yyval.fal_stat) = 0; // we don't add any statement
      }
      else {
         // create anyhow a nil property
          if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), new Falcon::VarDef() );
         // but also prepare a statement to be executed by the auto-constructor.
         (yyval.fal_stat) = new Falcon::StmtVarDef( LINE, (yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
   }
    break;

  case 292:

/* Line 1455 of yacc.c  */
#line 1984 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { 
         (yyval.fal_stat) = COMPILER->getContext(); 
         COMPILER->popContext();
      }
    break;

  case 293:

/* Line 1455 of yacc.c  */
#line 1992 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->pushContext( 
            new Falcon::StmtState( (yyvsp[(2) - (3)].stringp), 
                static_cast<Falcon::StmtClass*>( COMPILER->getContext() ) ) ); 
      }
    break;

  case 294:

/* Line 1455 of yacc.c  */
#line 1998 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass* cls = 
            static_cast<Falcon::StmtClass*>( COMPILER->getContext() );
         
         Falcon::StmtState* state = new Falcon::StmtState( COMPILER->addString( "init" ), cls );
         cls->initState( state );
         
         COMPILER->pushContext( state ); 
      }
    break;

  case 298:

/* Line 1455 of yacc.c  */
#line 2019 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 299:

/* Line 1455 of yacc.c  */
#line 2030 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
            sym->setEnum( true );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setClass( def );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         // We don't have a context set here
         COMPILER->pushFunction( def );

         COMPILER->resetEnum();
      }
    break;

  case 300:

/* Line 1455 of yacc.c  */
#line 2064 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      }
    break;

  case 304:

/* Line 1455 of yacc.c  */
#line 2081 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
    break;

  case 306:

/* Line 1455 of yacc.c  */
#line 2086 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (2)].stringp) );
      }
    break;

  case 309:

/* Line 1455 of yacc.c  */
#line 2101 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // we create a special symbol for the class.
         Falcon::String cl_name = "%";
         cl_name += *(yyvsp[(2) - (2)].stringp);
         Falcon::Symbol *clsym = COMPILER->addGlobalSymbol( COMPILER->addString( cl_name ) );
         clsym->setClass( def );

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setInstance( clsym );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), clsym );
         cls->singleton( sym );

         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         //Statements here goes in the auto constructor.
         //COMPILER->pushContextSet( &cls->autoCtor() );
         COMPILER->pushFunction( def );
      }
    break;

  case 310:

/* Line 1455 of yacc.c  */
#line 2141 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // check for expressions in from clauses
         COMPILER->checkLocalUndefined();

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( !cd->inheritance().empty() )
            {
               COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //COMPILER->popContextSet();
         COMPILER->popFunction();
      }
    break;

  case 312:

/* Line 1455 of yacc.c  */
#line 2169 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      }
    break;

  case 316:

/* Line 1455 of yacc.c  */
#line 2181 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 317:

/* Line 1455 of yacc.c  */
#line 2184 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      COMPILER->checkLocalUndefined();
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   }
    break;

  case 320:

/* Line 1455 of yacc.c  */
#line 2213 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      }
    break;

  case 321:

/* Line 1455 of yacc.c  */
#line 2218 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // raise an error if we are not in a local context
         if ( ! COMPILER->isLocalContext() )
         {
            COMPILER->raiseError(Falcon::e_global_notin_func, "", LINE );
         }
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
      }
    break;

  case 323:

/* Line 1455 of yacc.c  */
#line 2232 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 324:

/* Line 1455 of yacc.c  */
#line 2237 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 326:

/* Line 1455 of yacc.c  */
#line 2243 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 327:

/* Line 1455 of yacc.c  */
#line 2250 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      }
    break;

  case 328:

/* Line 1455 of yacc.c  */
#line 2265 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); }
    break;

  case 329:

/* Line 1455 of yacc.c  */
#line 2266 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 330:

/* Line 1455 of yacc.c  */
#line 2267 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; }
    break;

  case 331:

/* Line 1455 of yacc.c  */
#line 2275 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); }
    break;

  case 332:

/* Line 1455 of yacc.c  */
#line 2276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); }
    break;

  case 333:

/* Line 1455 of yacc.c  */
#line 2277 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); }
    break;

  case 334:

/* Line 1455 of yacc.c  */
#line 2278 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); }
    break;

  case 335:

/* Line 1455 of yacc.c  */
#line 2279 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); }
    break;

  case 336:

/* Line 1455 of yacc.c  */
#line 2280 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); }
    break;

  case 337:

/* Line 1455 of yacc.c  */
#line 2284 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); }
    break;

  case 338:

/* Line 1455 of yacc.c  */
#line 2285 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); }
    break;

  case 339:

/* Line 1455 of yacc.c  */
#line 2286 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); }
    break;

  case 340:

/* Line 1455 of yacc.c  */
#line 2287 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); }
    break;

  case 341:

/* Line 1455 of yacc.c  */
#line 2288 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); }
    break;

  case 342:

/* Line 1455 of yacc.c  */
#line 2289 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); }
    break;

  case 343:

/* Line 1455 of yacc.c  */
#line 2294 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *val;
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            val = new Falcon::Value();
            val->setSymdef( (yyvsp[(1) - (1)].stringp) );
            // warning: the symbol is still undefined.
            COMPILER->addSymdef( val );
         }
         else {
            val = new Falcon::Value( sym );
         }
         (yyval.fal_val) = val;
     }
    break;

  case 345:

/* Line 1455 of yacc.c  */
#line 2312 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); }
    break;

  case 346:

/* Line 1455 of yacc.c  */
#line 2313 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtFunction *sfunc = COMPILER->getFunctionContext();
      if ( sfunc == 0 ) {
         COMPILER->raiseError(Falcon::e_fself_outside, COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value();
      }
      else
      {
         (yyval.fal_val) = new Falcon::Value( sfunc->symbol() );
      }
   }
    break;

  case 351:

/* Line 1455 of yacc.c  */
#line 2341 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( (yyvsp[(2) - (2)].stringp) ); /* do not add the symbol to the compiler */ }
    break;

  case 352:

/* Line 1455 of yacc.c  */
#line 2342 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { char space[32]; sprintf(space, "%d", (int)(yyvsp[(2) - (2)].integer) ); (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(space) ); }
    break;

  case 353:

/* Line 1455 of yacc.c  */
#line 2343 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString("self") ); /* do not add the symbol to the compiler */ }
    break;

  case 354:

/* Line 1455 of yacc.c  */
#line 2344 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyvsp[(3) - (3)].stringp)->prepend( "." ); (yyval.fal_val)->setLBind( (yyvsp[(3) - (3)].stringp) ); /* do not add the symbol to the compiler */ }
    break;

  case 355:

/* Line 1455 of yacc.c  */
#line 2345 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { char space[32]; sprintf(space, ".%d", (int)(yyvsp[(3) - (3)].integer) ); (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(space) ); }
    break;

  case 356:

/* Line 1455 of yacc.c  */
#line 2346 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(".self") ); /* do not add the symbol to the compiler */ }
    break;

  case 357:

/* Line 1455 of yacc.c  */
#line 2347 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 358:

/* Line 1455 of yacc.c  */
#line 2348 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_fbind, new Falcon::Value((yyvsp[(1) - (3)].stringp)), (yyvsp[(3) - (3)].fal_val)) ); }
    break;

  case 359:

/* Line 1455 of yacc.c  */
#line 2349 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            // is this an immediate string sum ?
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isString() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str += *(yyvsp[(4) - (4)].fal_val)->asString();
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str.writeNumber( (yyvsp[(4) - (4)].fal_val)->asInteger() );
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
         }
    break;

  case 360:

/* Line 1455 of yacc.c  */
#line 2375 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 361:

/* Line 1455 of yacc.c  */
#line 2376 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( (yyvsp[(1) - (4)].fal_val)->asString()->length() );
                  for( int i = 0; i < (yyvsp[(4) - (4)].fal_val)->asInteger(); ++i )
                  {
                     str.append( *(yyvsp[(1) - (4)].fal_val)->asString()  );
                  }
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 362:

/* Line 1455 of yacc.c  */
#line 2396 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if( (yyvsp[(1) - (4)].fal_val)->asString()->length() == 0 )
               {
                  COMPILER->raiseError( Falcon::e_invop );
               }
               else {

                  if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
                  {
                     Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                     str.setCharAt( str.length()-1, str.getCharAt(str.length()-1) + (yyvsp[(4) - (4)].fal_val)->asInteger() );
                     (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                     delete (yyvsp[(4) - (4)].fal_val);
                     (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
                  }
                  else
                     (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
               }
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 363:

/* Line 1455 of yacc.c  */
#line 2420 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str.append( (Falcon::uint32) (yyvsp[(4) - (4)].fal_val)->asInteger() );
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 364:

/* Line 1455 of yacc.c  */
#line 2437 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 365:

/* Line 1455 of yacc.c  */
#line 2438 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 366:

/* Line 1455 of yacc.c  */
#line 2439 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 367:

/* Line 1455 of yacc.c  */
#line 2440 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 368:

/* Line 1455 of yacc.c  */
#line 2441 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 369:

/* Line 1455 of yacc.c  */
#line 2442 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 370:

/* Line 1455 of yacc.c  */
#line 2443 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 371:

/* Line 1455 of yacc.c  */
#line 2444 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 372:

/* Line 1455 of yacc.c  */
#line 2445 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 373:

/* Line 1455 of yacc.c  */
#line 2446 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 374:

/* Line 1455 of yacc.c  */
#line 2447 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 375:

/* Line 1455 of yacc.c  */
#line 2448 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 376:

/* Line 1455 of yacc.c  */
#line 2449 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 377:

/* Line 1455 of yacc.c  */
#line 2450 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 378:

/* Line 1455 of yacc.c  */
#line 2451 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 379:

/* Line 1455 of yacc.c  */
#line 2452 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 380:

/* Line 1455 of yacc.c  */
#line 2453 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 381:

/* Line 1455 of yacc.c  */
#line 2454 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 382:

/* Line 1455 of yacc.c  */
#line 2455 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 383:

/* Line 1455 of yacc.c  */
#line 2456 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 384:

/* Line 1455 of yacc.c  */
#line 2457 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 385:

/* Line 1455 of yacc.c  */
#line 2458 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 386:

/* Line 1455 of yacc.c  */
#line 2459 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); }
    break;

  case 387:

/* Line 1455 of yacc.c  */
#line 2460 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 388:

/* Line 1455 of yacc.c  */
#line 2461 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (Falcon::Value *) 0 ); }
    break;

  case 389:

/* Line 1455 of yacc.c  */
#line 2462 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 390:

/* Line 1455 of yacc.c  */
#line 2463 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 391:

/* Line 1455 of yacc.c  */
#line 2464 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eval, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 392:

/* Line 1455 of yacc.c  */
#line 2465 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_oob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 393:

/* Line 1455 of yacc.c  */
#line 2466 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_deoob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 394:

/* Line 1455 of yacc.c  */
#line 2467 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_isoob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 395:

/* Line 1455 of yacc.c  */
#line 2468 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_xoroob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 402:

/* Line 1455 of yacc.c  */
#line 2476 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 403:

/* Line 1455 of yacc.c  */
#line 2481 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].fal_adecl) );
   }
    break;

  case 404:

/* Line 1455 of yacc.c  */
#line 2485 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( exp );
   }
    break;

  case 405:

/* Line 1455 of yacc.c  */
#line 2490 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 406:

/* Line 1455 of yacc.c  */
#line 2496 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' && ! (yyvsp[(1) - (3)].fal_val)->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 409:

/* Line 1455 of yacc.c  */
#line 2508 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   }
    break;

  case 410:

/* Line 1455 of yacc.c  */
#line 2513 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyvsp[(5) - (5)].fal_adecl)->pushFront( (yyvsp[(3) - (5)].fal_val) );
      Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (5)].fal_adecl) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (5)].fal_val), second ) );
   }
    break;

  case 411:

/* Line 1455 of yacc.c  */
#line 2520 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aadd, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 412:

/* Line 1455 of yacc.c  */
#line 2521 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_asub, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 413:

/* Line 1455 of yacc.c  */
#line 2522 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amul, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 414:

/* Line 1455 of yacc.c  */
#line 2523 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_adiv, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 415:

/* Line 1455 of yacc.c  */
#line 2524 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amod, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 416:

/* Line 1455 of yacc.c  */
#line 2525 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_apow, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 417:

/* Line 1455 of yacc.c  */
#line 2526 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aband, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 418:

/* Line 1455 of yacc.c  */
#line 2527 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 419:

/* Line 1455 of yacc.c  */
#line 2528 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abxor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 420:

/* Line 1455 of yacc.c  */
#line 2529 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashl, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 421:

/* Line 1455 of yacc.c  */
#line 2530 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashr, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 422:

/* Line 1455 of yacc.c  */
#line 2531 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_val)=(yyvsp[(2) - (3)].fal_val);}
    break;

  case 423:

/* Line 1455 of yacc.c  */
#line 2536 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      }
    break;

  case 424:

/* Line 1455 of yacc.c  */
#line 2539 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      }
    break;

  case 425:

/* Line 1455 of yacc.c  */
#line 2542 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      }
    break;

  case 426:

/* Line 1455 of yacc.c  */
#line 2545 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      }
    break;

  case 427:

/* Line 1455 of yacc.c  */
#line 2548 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) ) );
      }
    break;

  case 428:

/* Line 1455 of yacc.c  */
#line 2555 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) ) );
      }
    break;

  case 429:

/* Line 1455 of yacc.c  */
#line 2561 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (3)].fal_val), 0 ) );
      }
    break;

  case 430:

/* Line 1455 of yacc.c  */
#line 2565 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 431:

/* Line 1455 of yacc.c  */
#line 2566 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(3) - (6)].fal_adecl);
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 432:

/* Line 1455 of yacc.c  */
#line 2575 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         COMPILER->incClosureContext();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
         COMPILER->lexer()->pushContext( Falcon::SrcLexer::ct_inner, COMPILER->lexer()->line() );
      }
    break;

  case 433:

/* Line 1455 of yacc.c  */
#line 2610 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->lexer()->popContext();
            (yyval.fal_val) = COMPILER->closeClosure();
         }
    break;

  case 434:

/* Line 1455 of yacc.c  */
#line 2618 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         COMPILER->incClosureContext();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      }
    break;

  case 435:

/* Line 1455 of yacc.c  */
#line 2652 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StatementList *stmt = COMPILER->getContextSet();
         if( stmt->size() == 1 && stmt->back()->type() == Falcon::Statement::t_autoexp )
         {
            // wrap it around a return, so A is not nilled.
            Falcon::StmtAutoexpr *ae = static_cast<Falcon::StmtAutoexpr *>( stmt->pop_back() );
            stmt->push_back( new Falcon::StmtReturn( 1, ae->value()->clone() ) );

            // we don't need the expression anymore.
            delete ae;
         }

         (yyval.fal_val) = COMPILER->closeClosure();
      }
    break;

  case 437:

/* Line 1455 of yacc.c  */
#line 2671 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 438:

/* Line 1455 of yacc.c  */
#line 2675 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 440:

/* Line 1455 of yacc.c  */
#line 2683 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 441:

/* Line 1455 of yacc.c  */
#line 2687 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 442:

/* Line 1455 of yacc.c  */
#line 2694 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
         COMPILER->lexer()->pushContext( Falcon::SrcLexer::ct_inner, COMPILER->lexer()->line() );
      }
    break;

  case 443:

/* Line 1455 of yacc.c  */
#line 2728 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->lexer()->popContext();
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            // analyze func in previous context.
            COMPILER->closeFunction();
         }
    break;

  case 444:

/* Line 1455 of yacc.c  */
#line 2744 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   }
    break;

  case 445:

/* Line 1455 of yacc.c  */
#line 2749 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 446:

/* Line 1455 of yacc.c  */
#line 2756 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 447:

/* Line 1455 of yacc.c  */
#line 2763 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 448:

/* Line 1455 of yacc.c  */
#line 2772 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); }
    break;

  case 449:

/* Line 1455 of yacc.c  */
#line 2774 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 450:

/* Line 1455 of yacc.c  */
#line 2778 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 451:

/* Line 1455 of yacc.c  */
#line 2785 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); }
    break;

  case 452:

/* Line 1455 of yacc.c  */
#line 2787 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 453:

/* Line 1455 of yacc.c  */
#line 2791 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 454:

/* Line 1455 of yacc.c  */
#line 2799 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); }
    break;

  case 455:

/* Line 1455 of yacc.c  */
#line 2800 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_ddecl) ); }
    break;

  case 456:

/* Line 1455 of yacc.c  */
#line 2802 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) );
      }
    break;

  case 457:

/* Line 1455 of yacc.c  */
#line 2809 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 458:

/* Line 1455 of yacc.c  */
#line 2810 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 459:

/* Line 1455 of yacc.c  */
#line 2814 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 460:

/* Line 1455 of yacc.c  */
#line 2815 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 463:

/* Line 1455 of yacc.c  */
#line 2822 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      }
    break;

  case 464:

/* Line 1455 of yacc.c  */
#line 2828 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      }
    break;

  case 465:

/* Line 1455 of yacc.c  */
#line 2835 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); }
    break;

  case 466:

/* Line 1455 of yacc.c  */
#line 2836 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (5)].fal_ddecl)->pushBack( (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (5)].fal_ddecl); }
    break;



/* Line 1455 of yacc.c  */
#line 7493 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined(yyoverflow) || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}



/* Line 1675 of yacc.c  */
#line 2840 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


